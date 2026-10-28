
#ifndef QUANT_ALLTOALL_HOST_H
#define QUANT_ALLTOALL_HOST_H

#include "operator_registry.h"
#include "quant_size_utils.h"
#include "catlass/detail/alignment.hpp"

class QuantAllToAllOperator : public CatccosOperator {
public:
    static bool CheckProblemShape(uint32_t rankSize, const CocTilingParams& cocTiling)
    {
        if (rankSize == 0 || rankSize > MAX_RANK_SIZE || cocTiling.m == 0 || cocTiling.n == 0 || cocTiling.k == 0) {
            return false;
        }
        // Reduce before multiplying by K to avoid overflowing the three-factor product.
        uint64_t remainder = static_cast<uint64_t>(cocTiling.m) * cocTiling.n % rankSize;
        return remainder * cocTiling.k % rankSize == 0;
    }

    void AllocateDeviceSpace(
        KernelParams& params, const CocTilingParams& cocTiling, uint32_t rankId, std::string dataFile) override
    {
        size_t inputSize = CheckedBufferSize({cocTiling.m, cocTiling.n, cocTiling.k, sizeof(uint16_t)});
        size_t dataSize = inputSize / sizeof(uint16_t);
        uint32_t rankSize = cocTiling.rankSize;
        CheckQuantBlockCount(dataSize, rankSize);
        size_t scaleSize = static_cast<size_t>(rankSize) * sizeof(float);
        size_t outputSize = dataSize;

        uint8_t* inputDevice = AllocateAndLoadBuffer(
            inputSize, dataFile, "/rank_" + std::to_string(rankId) + "_input.bin", dataSize, static_cast<float16_t>(2));

        uint8_t* scaleDevice = AllocateAndLoadBuffer(scaleSize, dataFile, "/scale.bin", rankSize, 1.0f);

        uint8_t* outputDevice;
        ACL_CHECK(aclrtMalloc((void**)(&outputDevice), outputSize, ACL_MEM_MALLOC_HUGE_FIRST));

        params.SetKernelParams(inputDevice, scaleDevice, outputDevice);
    }

    void WriteResultFile(
        const KernelParams& params, const CocTilingParams& cocTiling, uint32_t rankId, std::string dataFile) override
    {
        size_t outputSize = CheckedBufferSize({cocTiling.m, cocTiling.n, cocTiling.k});

        uint8_t* outputDevice = params.ptrC;
        uint8_t* outputHost;
        ACL_CHECK(aclrtMallocHost((void**)(&outputHost), outputSize));
        ACL_CHECK(aclrtMemcpy(outputHost, outputSize, outputDevice, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));
        WriteFile(dataFile + "/rank_" + std::to_string(rankId) + "_output.bin", outputHost, outputSize);

        ACL_CHECK(aclrtFreeHost(outputHost));
    }

    size_t GetWorkspaceSize(const CocTilingParams& cocTiling) override { return 0; }

    CocCommType GetActualKernelType(const CocTilingParams& cocTiling) override { return CocCommType::QUANT_ALLTOALL; }

    bool CheckCocTilingParams(uint32_t rankSize, const CocTilingParams& cocTiling) override
    {
        if (!CheckProblemShape(rankSize, cocTiling)) {
            return false;
        }
        auto blockCount = MAX_BLOCK_COUNT;
        uint32_t commInterval = cocTiling.commInterval;
        int32_t maxPeerMemPerRank = SHMEM_BUFF_BYTES / sizeof(uint16_t) / rankSize / blockCount;
        return CheckedBufferSize({cocTiling.m, cocTiling.n, cocTiling.k, commInterval, 32}) < maxPeerMemPerRank;
    }

private:
    template <typename T>
    uint8_t* AllocateAndLoadBuffer(
        size_t bufferSize, const std::string& dataFile, const std::string& fileSuffix, size_t defaultCount,
        T defaultValue)
    {
        uint8_t* devicePtr;
        ACL_CHECK(aclrtMalloc((void**)(&devicePtr), bufferSize, ACL_MEM_MALLOC_HUGE_FIRST));
        if (!dataFile.empty()) {
            uint8_t* hostPtr;
            ACL_CHECK(aclrtMallocHost((void**)(&hostPtr), bufferSize));
            ReadFileOrThrow(dataFile + fileSuffix, hostPtr, bufferSize);
            ACL_CHECK(aclrtMemcpy(devicePtr, bufferSize, hostPtr, bufferSize, ACL_MEMCPY_HOST_TO_DEVICE));
            ACL_CHECK(aclrtFreeHost(hostPtr));
        } else {
            std::vector<T> defaultData(defaultCount, defaultValue);
            ACL_CHECK(aclrtMemcpy(devicePtr, bufferSize, defaultData.data(), bufferSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }
        return devicePtr;
    }
};

REGISTER_OPERATOR("QuantAllToAll", QuantAllToAllOperator);

#endif // QUANT_ALLTOALL_HOST_H
