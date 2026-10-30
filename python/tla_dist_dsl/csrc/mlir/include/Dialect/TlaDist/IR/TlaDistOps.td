include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Dialect/TlaDist/IR/TlaDistDialect.td"

class TlaDist_Op<string mnemonic, list<Trait> traits = []>
    : Op<TlaDistDialect, mnemonic, traits>;

def TlaDist_RemoteCopyOp
    : TlaDist_Op<"remote_copy", [MemoryEffects<[MemRead, MemWrite]>, SameVariadicOperandSize]> {
  let summary = "Asynchronously copy a tile to or from a symmetric peer";
  let arguments = (ins AnyType:$dst, AnyType:$src, I32:$peer,
                       Optional<AnyType>:$ub_tensor, Optional<AnyType>:$flag_value,
                       StrAttr:$mode, StrAttr:$transport,
                       OptionalAttr<UnitAttr>:$atomic_add,
                       OptionalAttr<StrAttr>:$wqe_pipe);
  let assemblyFormat = [{
    $dst `,` $src `,` $peer (`,` $ub_tensor^ `,` $flag_value)? attr-dict
    `:` type($dst) `,` type($src) `,` type($peer)
    (`,` type($ub_tensor)^ `,` type($flag_value))?
  }];
}

def TlaDist_BarrierAllOp
    : TlaDist_Op<"barrier_all", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Synchronize all participating ranks";
  let assemblyFormat = "attr-dict";
}
