#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import math
import os
import pandas as pd


def reorder_best_result(test_file, best_file, output_file=None):
    # 读取 CSV 文件
    test_df = pd.read_csv(test_file)
    best_df = pd.read_csv(best_file)

    # 检查所需列是否都存在
    required_columns = ['M', 'K', 'N', 'Transpose A', 'Transpose B']
    for col in required_columns:
        if col not in test_df.columns:
            raise ValueError(f"{test_file} 中缺失列：{col}")
        if col not in best_df.columns:
            raise ValueError(f"{best_file} 中缺失列：{col}")

    # 构建字典，用于记录 test_shapes.csv 中每个 shape 和输入布局出现的顺序
    order_map = {}
    for idx, row in test_df.iterrows():
        key = tuple(row[col] for col in required_columns)
        # 若存在重复组合，这里仅记录第一次出现的顺序
        if key not in order_map:
            order_map[key] = idx

    # 为每个 shape 和输入布局分配排序键
    best_df['order'] = best_df.apply(
        lambda row: order_map.get(tuple(row[col] for col in required_columns), float('inf')), axis=1
    )

    # 检查是否有 best_result 中的行未能在 test_shapes 中找到对应组合
    missing = best_df[best_df['order'] == float('inf')]
    if not missing.empty:
        print("警告：在 best_result.csv 中发现以下 shape/布局组合未在 test_shapes.csv 中出现：")
        print(missing[required_columns].drop_duplicates())

    # 对 best_df 按照 order 排序，并删除辅助的 order 列
    best_df_sorted = best_df.sort_values(by='order').drop(columns=['order'])

    # 将结果保存到输出文件中（如果 output_file 为 None，则覆盖 best_file）
    if output_file is None:
        output_file = best_file
    best_df_sorted.to_csv(output_file, index=False)
    print(f"排序后的文件保存到：{output_file}")


def process_csv_files(folder_path):
    # 存放所有result.csv的完整路径
    result_csv_files = []
    # 使用 os.walk 递归查找所有子目录下的 result.csv 文件
    for root, dirs, files in os.walk(folder_path):
        if "result.csv" in files:
            result_csv_files.append(os.path.join(root, "result.csv"))

    if not result_csv_files:
        print("no result.csv file is found.")
        return

    df_list = []
    required_columns = [
        "Op",
        "M",
        "K",
        "N",
        "Transpose A",
        "Transpose B",
        "commInterval",
        "commTileM",
        "commBlockM",
        "commNpuSplit",
        "commDataSplit",
        "Time(us)",
    ]

    # 对每个文件检查所需要的列是否齐全
    for file in result_csv_files:
        try:
            df = pd.read_csv(file)
            if not all(col in df.columns for col in required_columns):
                print(f"file {file} required column missing")
                continue
            df_list.append(df)
        except Exception as e:
            print(f"read {file} error occurred: {e}")

    if not df_list:
        print("No valid CSV file was read")
        return

    # 合并所有CSV数据
    all_data = pd.concat(df_list, ignore_index=True)

    # 转换 Time(us) 为数值型（若有非数字数据则转为 NaN）
    all_data["Time(us)"] = pd.to_numeric(all_data["Time(us)"], errors="coerce")

    # 只有正的、有限的测量值才能参与最优 tiling 的比较。
    valid_times = all_data["Time(us)"].map(math.isfinite) & (all_data["Time(us)"] > 0)
    all_data = all_data.loc[valid_times]
    if all_data.empty:
        raise ValueError("No positive finite Time(us) measurements were found in result.csv files")

    # 不同转置布局具有不同的输入和性能结果，必须分别选取最优 tiling。
    idx = all_data.groupby(["Op", "M", "K", "N", "Transpose A", "Transpose B"])["Time(us)"].idxmin()
    ans = all_data.loc[idx].reset_index(drop=True)

    ans.to_csv("best_result.csv", index=False)


if __name__ == "__main__":
    cur_file = os.path.abspath(__file__)
    util_dir = os.path.dirname(cur_file)
    project_root = os.path.dirname(util_dir)
    folder = os.path.join(project_root, 'output', 'msprof')
    process_csv_files(folder)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('test_case_source', type=str)
    args = parser.parse_args()

    test_file = args.test_case_source
    best_file = 'best_result.csv'
    output_file = 'best_result_sorted.csv'  # 或者设为 None 直接覆盖 best_result.csv
    reorder_best_result(test_file, best_file, output_file)
