"""
.. _l-plot-exporter-recipes-onnx-exporter-dynpad:

torch.onnx.export and padding one dimension to a mulitple of a constant
=======================================================================

This is a frequent task which does not play well with dynamic shapes.
Let's see how to avoid using :func:`torch.cond`.

A model with a test
+++++++++++++++++++
"""

from onnx.reference import ReferenceEvaluator
from onnx_array_api.plotting.graphviz_helper import plot_dot
from onnx_diagnostic.helpers import max_diff
import torch


# %%
# We define a model padding to a multiple of a constant.


class PadToMultiple(torch.nn.Module):
    def __init__(
        self,
        multiple: int,
        dim: int = 0,
    ):
        super().__init__()
        self.dim_to_pad = dim
        self.multiple = multiple

    def forward(self, x):
        shape = x.shape
        dim = x.shape[self.dim_to_pad]
        next_dim = ((dim + self.multiple - 1) // self.multiple) * self.multiple
        to_pad = next_dim - dim
        pad = torch.zeros(
            (*shape[: self.dim_to_pad], to_pad, *shape[self.dim_to_pad + 1 :]), dtype=x.dtype
        )
        return torch.cat([x, pad], dim=self.dim_to_pad)


model = PadToMultiple(4, dim=1)

# %%
# Let's check it runs.
x = torch.randn((6, 7, 8))
y = model(x)
print(f"x.shape={x.shape}, y.shape={y.shape}")

# Let's check it runs on another example.
x2 = torch.randn((6, 8, 8))
y2 = model(x2)
print(f"x2.shape={x2.shape}, y2.shape={y2.shape}")

# %%
# Export
# ++++++
#
# Let's defined the dynamic shapes and checks it exports.

DYNAMIC = torch.export.Dim.DYNAMIC
ep = torch.export.export(
    model, (x,), dynamic_shapes=({0: DYNAMIC, 1: DYNAMIC, 2: DYNAMIC},), strict=False
)
print(ep)

# %%
# We can also inline the local function.

ep = torch.onnx.export(
    model, (x,), dynamic_shapes=({0: "batch", 1: "seq_len", 2: "num_frames"},), dynamo=True
)

# %%
# Let's save it.
ep.save("plot_exporter_recipes_oe_dynpad.onnx")

# %%
# Validation
# ++++++++++
#
# Let's validate the exported model a set of inputs.
ref = ReferenceEvaluator(ep.model_proto)
inputs = [
    torch.randn((6, 8, 8)),
    torch.randn((6, 7, 8)),
    torch.randn((5, 8, 17)),
    torch.randn((1, 24, 4)),
    torch.randn((3, 9, 11)),
]
for inp in inputs:
    expected = model(inp)
    got = ref.run(None, {"x": inp.numpy()})
    diff = max_diff(expected, got[0])
    print(f"diff with shape={inp.shape} -> {expected.shape}: discrepancies={diff['abs']}")

# %%
# And visually.

plot_dot(ep.model_proto)
