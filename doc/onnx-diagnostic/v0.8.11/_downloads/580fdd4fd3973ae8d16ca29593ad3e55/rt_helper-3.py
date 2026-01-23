import numpy as np
from onnx import TensorProto
import onnx.helper as oh
from onnx.checker import check_model
from onnx.numpy_helper import from_array
import matplotlib.pyplot as plt
from onnxruntime import InferenceSession, SessionOptions
from onnx_diagnostic.helpers.rt_helper import (
    js_profile_to_dataframe,
    plot_ort_profile_timeline,
)


def get_model():
    model_def0 = oh.make_model(
        oh.make_graph(
            [
                oh.make_node("Add", ["X", "init1"], ["X1"]),
                oh.make_node("Abs", ["X"], ["X2"]),
                oh.make_node("Add", ["X", "init3"], ["inter"]),
                oh.make_node("Mul", ["X1", "inter"], ["Xm"]),
                oh.make_node("Sub", ["X2", "Xm"], ["final"]),
            ],
            "test",
            [oh.make_tensor_value_info("X", TensorProto.FLOAT, [None])],
            [oh.make_tensor_value_info("final", TensorProto.FLOAT, [None])],
            [
                from_array(np.array([1], dtype=np.float32), name="init1"),
                from_array(np.array([3], dtype=np.float32), name="init3"),
            ],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=9,
    )
    check_model(model_def0)
    return model_def0


sess_options = SessionOptions()
sess_options.enable_profiling = True
sess = InferenceSession(
    get_model().SerializeToString(), sess_options, providers=["CPUExecutionProvider"]
)
for _ in range(11):
    sess.run(None, dict(X=np.arange(10).astype(np.float32)))
prof = sess.end_profiling()

df = js_profile_to_dataframe(prof, first_it_out=True)
print(df.head())

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
plot_ort_profile_timeline(df, ax, title="test_timeline", quantile=0.5)
fig.tight_layout()