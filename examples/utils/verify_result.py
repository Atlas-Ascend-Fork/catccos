import sys
import traceback
import os
from enum import Enum
import numpy as np
import scipy
import torch
import logging
import random
from utils import DataType, tensor_from_file, get_rtol

RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"
HIF8_DTYPE_NAMES = {"hif8", "hifloat8"}


class OpTypes(Enum):
    NA = 0  # new standard is not available
    MOVE = 1
    RAND = 2
    CAST = 3
    COMPUTE_INTEGER = 4
    COMPUTE_QUANT = 5
    COMPUTE_FLOAT = 6
    COMPUTE_FLOAT_HIGH_PRECISION = 7
    VECTOR_FUSION = 8
    CV_FUSION = 9


def parse_output_dtype(arg: str):
    normalized = arg.strip().lower()
    if normalized in HIF8_DTYPE_NAMES:
        return normalized
    return DataType.from_str(arg)


def is_hif8_dtype(dtype):
    return isinstance(dtype, str) and dtype in HIF8_DTYPE_NAMES


def load_tensor_for_verify(file_name: str, dtype):
    if is_hif8_dtype(dtype):
        from en_dtypes import hifloat8

        return torch.from_numpy(np.fromfile(file_name, dtype=hifloat8).astype(np.float32))
    return tensor_from_file(file_name, dtype=dtype)


def get_precision_and_eb_threshold(op_type, dtype, rtol: float = 2 ** (-8)):
    precision_threshold = 0
    eb_threshold = 0
    if op_type in [OpTypes.MOVE, OpTypes.RAND, OpTypes.CAST, OpTypes.COMPUTE_INTEGER]:
        pass
    if op_type in [OpTypes.COMPUTE_QUANT]:
        if dtype in [torch.int8]:
            precision_threshold = 1
    if op_type in [OpTypes.COMPUTE_QUANT, OpTypes.COMPUTE_FLOAT]:
        if dtype in [torch.float16]:
            precision_threshold = rtol
            eb_threshold = 2 ** (-10)
        if dtype in [torch.bfloat16]:
            precision_threshold = rtol
            eb_threshold = 2 ** (-7)
        if dtype in [torch.float32]:
            precision_threshold = rtol
            eb_threshold = 2 ** (-14)
    if op_type in [OpTypes.COMPUTE_FLOAT_HIGH_PRECISION]:
        if dtype in [torch.float16]:
            precision_threshold = 2 ** (-11)
            eb_threshold = 2 ** (-10)
        if dtype in [torch.bfloat16]:
            precision_threshold = 2 ** (-8)
            eb_threshold = 2 ** (-7)
        if dtype in [torch.float32]:
            precision_threshold = 2 ** (-14)
            eb_threshold = 2 ** (-14)
    if op_type in [OpTypes.VECTOR_FUSION]:
        if dtype in [torch.float16]:
            precision_threshold = 2 ** (-9)
            eb_threshold = 2 ** (-10)
        if dtype in [torch.bfloat16]:
            precision_threshold = 2 ** (-8)
            eb_threshold = 2 ** (-7)
        if dtype in [torch.float32]:
            precision_threshold = 2 ** (-12)
            eb_threshold = 2 ** (-14)
    if op_type in [OpTypes.CV_FUSION]:
        precision_threshold = 522  # 最大相对误差5/平均相对误差2/均方根误差2
        if dtype in [torch.float16]:
            eb_threshold = 2 ** (-10)
        if dtype in [torch.bfloat16]:
            eb_threshold = 2 ** (-7)
        if dtype in [torch.float32]:
            eb_threshold = 2 ** (-14)
    logging.debug(
        f"op_type: {op_type}, dtype: {dtype}, precision_threshold: {precision_threshold}, eb_threshold: {eb_threshold}"
    )
    return precision_threshold, eb_threshold


def precision_performance_analysis(op_type, golden_output_tensor_list, output_tensor_list, rtol: float):
    for i in range(len(golden_output_tensor_list)):
        actual_output = output_tensor_list[i].cpu()
        golden_output = golden_output_tensor_list[i]
        precision_threshold, eb_threshold = get_precision_and_eb_threshold(op_type, actual_output.dtype, rtol)
        precision, eb = cal_precision_eb_percent(
            op_type, i, actual_output, golden_output, precision_threshold, eb_threshold
        )
    if precision == 100 and eb <= 100:
        return True
    else:
        print(f"precision: {precision}, eb: {eb}")
        return False


def double_precision_performance_analysis(
    op_type, golden_output_tensor_list, golden_low_output_tensor_list, output_tensor_list, rtol: float
):
    for i in range(len(golden_output_tensor_list)):
        actual_output = output_tensor_list[i].cpu()
        golden_output = golden_output_tensor_list[i]
        golden_low_output = golden_low_output_tensor_list[i]
        precision_threshold, eb_threshold = get_precision_and_eb_threshold(op_type, actual_output.dtype, rtol)
        max_relative_error, mean_relative_error, rmse, eb = cal_double_precision_eb_percent(
            op_type, i, actual_output, golden_output, golden_low_output, precision_threshold, eb_threshold
        )
    if max_relative_error < 10 and mean_relative_error < 2 and rmse < 2 and eb <= 100:
        return True
    else:
        print(
            f"max_relative_error: {max_relative_error}, mean_relative_error: {mean_relative_error}, rmse: {rmse}, eb: {eb}"
        )
        return False


def show_random_samples(actual, golden, num_samples=100):
    if actual.shape != golden.shape:
        raise ValueError("Shape mismatch")
    total = actual.numel()
    indices = random.SystemRandom().sample(range(total), min(num_samples, total))

    print(f"\n🎲 随机抽取 {len(indices)} 个位置的数据对比：")
    print(f"{'Index':>6} | {'Actual':>12} | {'Golden':>12} | {'AbsErr':>12} | {'RelErr':>12}")
    print("-" * 60)
    for i in indices:
        actual_value = actual[i].item()
        golden_value = golden[i].item()
        abs_err = abs(actual_value - golden_value)
        rel_err = abs_err / (abs(golden_value) + 1e-8)
        print(f"{i:>6} | {actual_value:>12.6f} | {golden_value:>12.6f} | {abs_err:>12.6f} | {rel_err:>12.6f}")


def safe_relative_error(actual_value, golden_value, eps=1e-8):
    abs_err = abs(actual_value - golden_value)
    denom = abs(golden_value)
    if denom == 0:
        return 0.0 if abs_err == 0 else abs_err / eps
    return abs_err / denom


def show_random_samples_double_fp16(t_fp16_0, t_fp16_1, t_fp32, num_samples=100):
    if t_fp16_0.shape != t_fp32.shape:
        raise ValueError("Shape0 mismatch")
    if t_fp16_1.shape != t_fp32.shape:
        raise ValueError("Shape1 mismatch")
    total = t_fp16_0.numel()
    indices = random.SystemRandom().sample(range(total), min(num_samples, total))

    print(f"\n🎲 随机抽取 {len(indices)} 个位置的数据对比：")
    print(f"{'Index':>6} | {'FP16_0':>12} | {'FP16_1':>12} | {'FP32':>12} | {'AbsErr':>12} | {'RelErr':>12}")
    print("-" * 75)
    for i in indices:
        val_fp16_0 = t_fp16_0[i].item()
        val_fp16_1 = t_fp16_1[i].item()
        val_fp32 = t_fp32[i].item()
        abs_err_0 = abs(val_fp16_0 - val_fp32)
        rel_err_0 = abs_err_0 / (abs(val_fp32) + 1e-8)
        print(
            f"{i:>6} | {val_fp16_0:>12.6f} | {val_fp16_1:>12.6f} | {val_fp32:>12.6f} | {abs_err_0:>12.6f} | {rel_err_0:>12.6f}"
        )


def count_nonfinite_mismatches(actual_output, golden_output):
    return torch.count_nonzero(torch.isfinite(actual_output) != torch.isfinite(golden_output)).item()


def cal_precision_eb_percent(op_type, i, actual_output, golden_output, precision_threshold, eb_threshold):
    actual_output = actual_output if actual_output.dtype != torch.bool else actual_output.long()
    golden_output = golden_output if golden_output.dtype != torch.bool else golden_output.long()
    if op_type in [
        OpTypes.COMPUTE_FLOAT,
        OpTypes.COMPUTE_FLOAT_HIGH_PRECISION,
        OpTypes.VECTOR_FUSION,
    ] and actual_output.dtype in [torch.float16, torch.bfloat16]:
        actual_output = actual_output.to(torch.float32)
        golden_output = golden_output.to(torch.float32)
    mismatch_count = count_nonfinite_mismatches(actual_output, golden_output)
    if mismatch_count:
        print(f"non-finite mismatch num: {mismatch_count}")
        return 0, 0
    # 对于输出中出现的NAN以及INF全部替换成0
    actual_output = torch.where(torch.isnan(actual_output), torch.full_like(actual_output, 0), actual_output)
    actual_output = torch.where(torch.isinf(actual_output), torch.full_like(actual_output, 0), actual_output)
    golden_output = torch.where(torch.isnan(golden_output), torch.full_like(golden_output, 0), golden_output)
    golden_output = torch.where(torch.isinf(golden_output), torch.full_like(golden_output, 0), golden_output)
    if op_type == OpTypes.RAND:
        alpha = 0.01
        t_statistic, p_value = scipy.stats.ks_2samp(actual_output, golden_output)
        precision_percent = 100 if p_value > alpha else 0
        eb_percent = 0
        return precision_percent, eb_percent
    diff = torch.subtract(actual_output, golden_output)
    tensor_max = torch.maximum(torch.ones(golden_output.shape, dtype=golden_output.dtype), torch.abs(golden_output))
    if precision_threshold == 1:
        tolerance = torch.subtract(torch.abs(diff), torch.ones(diff.shape, dtype=diff.dtype))
    else:
        tolerance = torch.subtract(torch.abs(diff), precision_threshold * tensor_max)

    different_element_indexes = torch.where(tolerance > 0)[0]
    for index in range(len(different_element_indexes)):
        real_index = different_element_indexes[index]
        golden_data = golden_output[real_index].item()
        output_data = actual_output[real_index].item()
        rdiff = safe_relative_error(output_data, golden_data)
        print(
            f"data index: {real_index:6d}, expected: {golden_data:-.9f}, actual: {output_data:-.9f}, "
            f"rdiff: {rdiff:-.6f}"
        )
        if index == 10:
            break
    error_num = len(different_element_indexes)
    print(f"error num: {error_num}")

    # eb 统计误差偏移情况
    eb = eb_threshold
    if eb_threshold != 0:
        eb = torch.abs(torch.mean(torch.div(diff, tensor_max)))
    precision_percent = torch.sum(tolerance <= 0).numpy() / torch.numel(tolerance) * 100
    eb_percent = 0 if eb == 0 else torch.sum(eb).to(torch.float).numpy() / eb_threshold * 100
    return precision_percent, eb_percent


def cal_double_precision_eb_percent(
    op_type, i, actual_output, golden_output, golden_low_output, err_threshold, eb_threshold
):
    epsilon = 1e-7

    actual_output = actual_output.to(torch.float32)
    golden_output = golden_output.to(torch.float32)
    golden_low_output = golden_low_output.to(torch.float32)

    mismatch_count = count_nonfinite_mismatches(actual_output, golden_output)
    if mismatch_count:
        print(f"non-finite mismatch num: {mismatch_count}")
        return float("inf"), float("inf"), float("inf"), 0

    actual_output = torch.where(torch.isnan(actual_output), torch.full_like(actual_output, 0), actual_output)
    actual_output = torch.where(torch.isinf(actual_output), torch.full_like(actual_output, 0), actual_output)
    golden_output = torch.where(torch.isnan(golden_output), torch.full_like(golden_output, 0), golden_output)
    golden_output = torch.where(torch.isinf(golden_output), torch.full_like(golden_output, 0), golden_output)
    golden_low_output = torch.where(
        torch.isnan(golden_low_output), torch.full_like(golden_low_output, 0), golden_low_output
    )
    golden_low_output = torch.where(
        torch.isinf(golden_low_output), torch.full_like(golden_low_output, 0), golden_low_output
    )
    show_random_samples_double_fp16(actual_output, golden_low_output, golden_output)

    diff_actual = torch.subtract(actual_output, golden_output)
    diff_golden_low = torch.subtract(golden_low_output, golden_output)

    # eb 统计误差偏移情况
    tensor_max = torch.maximum(torch.ones(golden_output.shape, dtype=golden_output.dtype), torch.abs(golden_output))
    eb = eb_threshold
    if eb_threshold != 0:
        eb = torch.abs(torch.mean(torch.div(diff_actual, tensor_max)))
    eb_percent = 0 if eb == 0 else torch.sum(eb).to(torch.float).numpy() / eb_threshold * 100

    relative_error_actual = torch.abs(diff_actual) / (torch.abs(golden_output) + epsilon)
    relative_error_golden_low = torch.abs(diff_golden_low) / (torch.abs(golden_output) + epsilon)

    max_relative_error_actual = torch.max(relative_error_actual)
    max_relative_error_golden_low = torch.max(relative_error_golden_low)
    mean_relative_error_actual = torch.mean(relative_error_actual)
    mean_relative_error_golden_low = torch.mean(relative_error_golden_low)

    # 计算均方根误差
    mse_actual = torch.mean(diff_actual**2)
    rmse_actual = torch.sqrt(mse_actual)
    mse_golden_low = torch.mean(diff_golden_low**2)
    rmse_golden_low = torch.sqrt(mse_golden_low)

    print("max_relative_error_catccos:", max_relative_error_actual)
    print("max_relative_error_golden_low:", max_relative_error_golden_low)
    print("mean_relative_error_catccos:", mean_relative_error_actual)
    print("mean_relative_error_golden_low:", mean_relative_error_golden_low)
    print("rmse_catccos:", rmse_actual)
    print("rmse_golden_low:", rmse_golden_low)

    return (
        max_relative_error_actual / max(max_relative_error_golden_low, err_threshold),
        mean_relative_error_actual / max(mean_relative_error_golden_low, err_threshold),
        rmse_actual / max(rmse_golden_low, err_threshold),
        eb_percent,
    )


def verify_result():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=str)
    parser.add_argument('golden', type=str)
    parser.add_argument('out_dtype', type=parse_output_dtype)
    parser.add_argument('m', type=int)
    parser.add_argument('n', type=int)
    parser.add_argument('k', type=int)
    parser.add_argument('--golden_low', default="", type=str, help="low-precision golden bin file")
    args = parser.parse_args()

    if is_hif8_dtype(args.out_dtype):
        golden = load_tensor_for_verify(args.golden, args.out_dtype)
        output = load_tensor_for_verify(args.output, args.out_dtype)
        rtol = get_rtol(dtype=torch.float32, compute_times=args.k)
    else:
        output = tensor_from_file(args.output, dtype=args.out_dtype.torch_type).reshape(-1, args.n).flatten()
        output_elem_size = torch.empty((), dtype=args.out_dtype.torch_type).element_size()
        if os.path.getsize(args.golden) == output.numel() * output_elem_size:
            golden = tensor_from_file(args.golden, dtype=args.out_dtype.torch_type)
        else:
            golden = tensor_from_file(args.golden, dtype=torch.float32)
        output = output[: golden.shape[0]]
        rtol = get_rtol(dtype=args.out_dtype.torch_type, compute_times=args.k)

    show_random_samples(output.cpu().to(torch.float32), golden.cpu().to(torch.float32))

    if args.golden_low != "":
        golden_low_dtype = args.out_dtype if is_hif8_dtype(args.out_dtype) else args.out_dtype.torch_type
        golden_low = load_tensor_for_verify(args.golden_low, golden_low_dtype)
        result = double_precision_performance_analysis(
            OpTypes.COMPUTE_FLOAT_HIGH_PRECISION, [golden.cpu()], [golden_low.cpu()], [output.cpu()], rtol
        )
    else:
        result = precision_performance_analysis(OpTypes.COMPUTE_FLOAT, [golden.cpu()], [output.cpu()], rtol)
    return result


if __name__ == '__main__':
    try:
        res = verify_result()
        if not res:
            print(f"{RED}ERROR{RESET}")
            sys.exit(1)
        else:
            print(f"{GREEN}PASS{RESET}")
        sys.exit(0 if res else 1)
    except Exception as e:
        print(e)
        traceback.print_exc()
        sys.exit(1)
