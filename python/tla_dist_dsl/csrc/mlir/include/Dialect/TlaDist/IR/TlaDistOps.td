// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This file is a part of the CANN Open Software.
// Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Dialect/TlaDist/IR/TlaDistDialect.td"

class TlaDist_Op<string mnemonic, list<Trait> traits = []>
    : Op<TlaDistDialect, mnemonic, traits>;

def TlaDist_RemoteCopyOp
    : TlaDist_Op<"remote_copy", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Asynchronously copy a tile to or from a symmetric peer";
  let arguments = (ins AnyType:$dst, AnyType:$src, I32:$peer, I32:$qp_idx,
                       Optional<AnyType>:$ub_tensor,
                       OptionalAttr<ConfinedAttr<I32Attr, [IntMinValue<0>, IntMaxValue<7>]>>:$event_id,
                       StrAttr:$mode, StrAttr:$transport,
                       OptionalAttr<UnitAttr>:$atomic_add,
                       OptionalAttr<StrAttr>:$wqe_pipe);
  let assemblyFormat = [{
    $dst `,` $src `,` $peer `,` $qp_idx (`,` $ub_tensor^)? attr-dict
    `:` type($dst) `,` type($src) `,` type($peer) `,` type($qp_idx)
    (`,` type($ub_tensor)^)?
  }];
}

def TlaDist_QuietOp
    : TlaDist_Op<"quiet", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Wait for requests submitted to one peer";
  let arguments = (ins I32:$peer, I32:$qp_idx, Optional<AnyType>:$ub_tensor,
                       OptionalAttr<ConfinedAttr<I32Attr, [IntMinValue<0>, IntMaxValue<7>]>>:$event_id,
                       StrAttr:$transport);
  let assemblyFormat = [{
    $peer `,` $qp_idx (`,` $ub_tensor^)? attr-dict `:` type($peer) `,` type($qp_idx)
    (`,` type($ub_tensor)^)?
  }];
}

def TlaDist_BarrierAllOp
    : TlaDist_Op<"barrier_all", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Synchronize all participating ranks";
  let assemblyFormat = "attr-dict";
}
