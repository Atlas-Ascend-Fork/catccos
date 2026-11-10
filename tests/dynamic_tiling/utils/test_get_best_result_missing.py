#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from shutil import copyfile


SCRIPT = Path(__file__).with_name("get_best_result.py")
HEADER = "Op,M,K,N,Transpose A,Transpose B,commInterval,commTileM,commBlockM,commNpuSplit,commDataSplit,Time(us)\n"
ROW = "mmar,64,64,64,0,0,1,64,64,1,1,123.0\n"


class BestResultInputTest(unittest.TestCase):
    def run_case(self, result_contents):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            utils = root / "utils"
            utils.mkdir()
            copyfile(SCRIPT, utils / SCRIPT.name)
            (root / "best_result.csv").write_text(HEADER + ROW.replace("123.0", "999.0"), encoding="utf-8")
            shapes = root / "test_shapes.csv"
            shapes.write_text("M,K,N,Transpose A,Transpose B\n64,64,64,0,0\n", encoding="utf-8")
            if result_contents is not None:
                result = root / "output" / "msprof" / "result.csv"
                result.parent.mkdir(parents=True)
                result.write_text(result_contents, encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(utils / SCRIPT.name), str(shapes)],
                cwd=root,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            sorted_file = root / "best_result_sorted.csv"
            return proc, sorted_file.read_text(encoding="utf-8") if sorted_file.exists() else None

    def test_missing_or_invalid_results_do_not_sort_stale_file(self):
        for contents in (None, "not_a_valid_column\n1\n"):
            with self.subTest(contents=contents):
                proc, sorted_data = self.run_case(contents)
                self.assertNotEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                self.assertIsNone(sorted_data)

    def test_valid_result_is_sorted(self):
        proc, sorted_data = self.run_case(HEADER + ROW)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("123.0", sorted_data)
        self.assertNotIn("999.0", sorted_data)


if __name__ == "__main__":
    unittest.main()
