include "mlir/IR/EnumAttr.td"

def TlaDist_CoreScope : I32EnumAttr<"CoreScope", "local core participation", [
    I32EnumAttrCase<"AIV", 0>, I32EnumAttrCase<"MIX", 1>]> {
  let cppNamespace = "::tla_dist";
  let genSpecializedAttr = 1;
}
def TlaDist_SignalKind : I32EnumAttr<"SignalKind", "signal update operation", [
    I32EnumAttrCase<"SET", 0>, I32EnumAttrCase<"ADD", 1>]> {
  let cppNamespace = "::tla_dist";
  let genSpecializedAttr = 1;
}
def TlaDist_CompareOp : I32EnumAttr<"CompareKind", "typed comparison", [
    I32EnumAttrCase<"EQ", 0>, I32EnumAttrCase<"NE", 1>,
    I32EnumAttrCase<"GT", 2>, I32EnumAttrCase<"GE", 3>,
    I32EnumAttrCase<"LT", 4>, I32EnumAttrCase<"LE", 5>]> {
  let cppNamespace = "::tla_dist";
  let genSpecializedAttr = 1;
}
