

#ifndef MX_QUANT_ALLGATHER_HOST_H
#define MX_QUANT_ALLGATHER_HOST_H

#include "operator_registry.h"
#include "quant_size_utils.h"
#include "catlass/detail/alignment.hpp"

class MxQuantAllGatherOperator : public CatccosOperator {
public:
    // BLOCK_SIZE for MX quantization (must match device template)
    static constexpr uint32_t MX_BLOCK_SIZE = 32;
    // Pack ratio: fp8=1 (1 byte/elem), fp4=2 (0.5 bytes/elem)
#if MX_QUANT_DTYPE >= 2
    static constexpr uint32_t PACK_RATIO = 2;
#else
    static constexpr uint32_t PACK_RATIO = 1;
#endif

    void AllocateDeviceSpace(
        KernelParams& params, const CocTilingParams& cocTiling, uint32_t rankId, std::string dataFile) override
    {
        uint32_t M = cocTiling.m;
        uint32_t N = cocTiling.n;
        uint32_t rankSize = cocTiling.rankSize;

        // Input: M × N bf16 elements (per rank)
        size_t inputSize = CheckedBufferSize({M, N, sizeof(uint16_t)});
        size_t dataSize = inputSize / sizeof(uint16_t);

        // Output: M × (N / PACK_RATIO) × rankSize quantized bytes
        size_t outputSize = CheckedBufferSize({M, N / PACK_RATIO, rankSize});

        // MxScale: M × (N / BLOCK_SIZE) × rankSize E8M0 bytes
        uint32_t numScalesPerRow = N / MX_BLOCK_SIZE;
        // N / MX_BLOCK_SIZE <= N / PACK_RATIO, so this is bounded by the checked outputSize.
        size_t mxScaleSize = static_cast<size_t>(M) * numScalesPerRow * rankSize;

        // Allocate input
        uint8_t* inputDevice = AllocateAndLoadBuffer(
            inputSize, dataFile, "/rank_" + std::to_string(rankId) + "_input.bin", dataSize,
            static_cast<fp16_t>(rankId + 1));

        // Allocate output (quantized data)
        uint8_t* outputDevice;
        ACL_CHECK(aclrtMalloc((void**)(&outputDevice), outputSize, ACL_MEM_MALLOC_HUGE_FIRST));

        // Allocate mxscale output
        uint8_t* mxScaleDevice;
        ACL_CHECK(aclrtMalloc((void**)(&mxScaleDevice), mxScaleSize, ACL_MEM_MALLOC_HUGE_FIRST));

        params.SetKernelParams(inputDevice, mxScaleDevice, outputDevice);
    }

    void WriteResultFile(
        const KernelParams& params, const CocTilingParams& cocTiling, uint32_t rankId, std::string dataFile) override
    {
        uint32_t M = cocTiling.m;
        uint32_t N = cocTiling.n;
        uint32_t rankSize = cocTiling.rankSize;
        uint32_t numScalesPerRow = N / MX_BLOCK_SIZE;

        // Write quantized output
        size_t outputSize = CheckedBufferSize({M, N / PACK_RATIO, rankSize});
        uint8_t* outputDevice = params.ptrC; // ptrC = outputDevice
        uint8_t* outputHost;
        ACL_CHECK(aclrtMallocHost((void**)(&outputHost), outputSize));
        ACL_CHECK(aclrtMemcpy(outputHost, outputSize, outputDevice, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));
        WriteFile(dataFile + "/rank_" + std::to_string(rankId) + "_output.bin", outputHost, outputSize);
        ACL_CHECK(aclrtFreeHost(outputHost));

        // Write mxscale output
        // Bounded by the checked outputSize, as in AllocateDeviceSpace.
        size_t mxScaleSize = static_cast<size_t>(M) * numScalesPerRow * rankSize;
        uint8_t* mxScaleDevice = params.ptrB; // ptrB = mxScaleDevice
        uint8_t* mxScaleHost;
        ACL_CHECK(aclrtMallocHost((void**)(&mxScaleHost), mxScaleSize));
        ACL_CHECK(aclrtMemcpy(mxScaleHost, mxScaleSize, mxScaleDevice, mxScaleSize, ACL_MEMCPY_DEVICE_TO_HOST));
        WriteFile(dataFile + "/rank_" + std::to_string(rankId) + "_mxscale.bin", mxScaleHost, mxScaleSize);
        ACL_CHECK(aclrtFreeHost(mxScaleHost));
    }

    size_t GetWorkspaceSize(const CocTilingParams& cocTiling) override { return 0; }

    CocCommType GetActualKernelType(const CocTilingParams& cocTiling) override
    {
        // Reuse QUANT_ALLGATHER or add a new enum entry
        return CocCommType::MX_QUANT_ALLGATHER;
    }

    bool CheckCocTilingParams(uint32_t rankSize, const CocTilingParams& cocTiling) override
    {
        auto blockCount = MAX_BLOCK_COUNT;
        uint32_t N = cocTiling.n;
        // Check N is divisible by BLOCK_SIZE
        if (N % MX_BLOCK_SIZE != 0)
            return false;
        // IPC buffer needs: quantized data + mxscale per block
        int32_t maxPeerMemPerRank = SHMEM_BUFF_BYTES / rankSize / blockCount;
        size_t bytesPerBlock = CheckedBufferSize({cocTiling.m, static_cast<size_t>(N) + N / MX_BLOCK_SIZE});
        return bytesPerBlock < maxPeerMemPerRank;
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

REGISTER_OPERATOR("MxQuantAllGather", MxQuantAllGatherOperator);

#endif // MX_QUANT_ALLGATHER_HOST_H
