import numpy as np
from onnx import TensorProto
from onnx_array_api.light_api import start
from onnx_array_api.plotting.graphviz_helper import plot_dot

def mk(shape):
    return np.array(shape, dtype=np.int64)

model = (
    start(opset=18, ir_version=9)
    .cst(mk([2, 2, 1024, 256]), "shape")
    .cst(mk([0]), "c0")
    .cst(mk([256]), "c256")
    .cst(mk([512]), "c512")
    .cst(mk([3]), "c3")
    .vin("X", TensorProto.FLOAT, ("a", "b", "c", "d"))
    .bring("shape")
    .ConstantOfShape()
    .rename("C1")
    .bring("shape")
    .ConstantOfShape()
    .rename("C2")
    .bring("X", "c256", "c512", "c3")
    .Slice()
    .rename("S1")
    .bring("C1", "S1")
    .Concat(axis=3)
    .rename("P1")
    .bring("X", "c0", "c256", "c3")
    .Slice()
    .Neg()
    .rename("S2")
    .bring("C1", "S2")
    .Concat(axis=3)
    .rename("P2")
    .bring("P1", "P2")
    .Add()
    .rename("Y")
    .vout(TensorProto.FLOAT, ("a", "b", "c", "d"))
    .to_onnx()
)
ax = plot_dot(model)
ax.set_title("Dummy graph")
plt.show()