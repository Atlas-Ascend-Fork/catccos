# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Expected-hang sync cases. Supervisor must enforce an external process timeout.

Exit 124 from a bounded timeout is the expected observation; these cases never
print DEVICE_PASS and are excluded from the successful acceptance manifest.
"""

import argparse
import os

from catccos import tla_dist  # isort: skip
from catlass import tla


@tla.kernel
def unsatisfied(counter: tla.Tensor):
    slot = tla.tile_view(counter, tla.make_shape(1, 1), tla.make_coord(0, 0))
    with tla.vector():
        tla_dist.wait_until(slot, 1)


@tla.kernel
def missing_participant():
    with tla.vector():
        tla_dist.sync()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=["wait-never", "missing-rank"])
    args = parser.parse_args()
    if os.environ.get("WORLD_SIZE") != "2":
        raise SystemExit("use two torchrun ranks with an external timeout")
    from test_remote_copy_mte import initialized_runtime

    with initialized_runtime() as runtime:
        if args.case == "wait-never":
            counter = runtime.shmem.symmetric_tensor((1, 16), runtime.torch.int32)
            counter.zero_()
            view = tla_dist.from_dlpack(counter, layout_tag=tla.arch.RowMajor)
            artifact = tla_dist.compile(unsatisfied, view, cache_dir=runtime.cache_dir)
            launch_args = (view,)
        else:
            artifact = tla_dist.compile(missing_participant, type_args=(), cache_dir=runtime.cache_dir)
            launch_args = ()
        runtime.torch.npu.synchronize()
        runtime.shmem.barrier_all()
        print(f"EXPECTED_HANG {args.case} rank={runtime.rank}", flush=True)
        if args.case == "wait-never" or runtime.rank == 0:
            artifact(*launch_args, block_num=1)
            runtime.torch.npu.synchronize()
            print(f"NEGATIVE_COMPLETED {args.case} rank={runtime.rank}", flush=True)
            raise RuntimeError("negative kernel returned unexpectedly")
        runtime.dist.barrier()


if __name__ == "__main__":
    main()
