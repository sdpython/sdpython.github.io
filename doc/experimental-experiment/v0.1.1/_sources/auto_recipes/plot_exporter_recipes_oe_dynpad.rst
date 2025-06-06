
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_recipes/plot_exporter_recipes_oe_dynpad.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_recipes_plot_exporter_recipes_oe_dynpad.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_recipes_plot_exporter_recipes_oe_dynpad.py:


.. _l-plot-exporter-recipes-onnx-exporter-dynpad:

torch.onnx.export and padding one dimension to a mulitple of a constant
=======================================================================

This is a frequent task which does not play well with dynamic shapes.
Let's see how to avoid using :func:`torch.cond`.

A model with a test
+++++++++++++++++++

.. GENERATED FROM PYTHON SOURCE LINES 13-20

.. code-block:: Python


    from onnx.reference import ReferenceEvaluator
    from onnx_array_api.plotting.graphviz_helper import plot_dot
    from onnx_diagnostic.helpers import max_diff
    import torch









.. GENERATED FROM PYTHON SOURCE LINES 21-22

We define a model padding to a multiple of a constant.

.. GENERATED FROM PYTHON SOURCE LINES 22-47

.. code-block:: Python



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








.. GENERATED FROM PYTHON SOURCE LINES 48-49

Let's check it runs.

.. GENERATED FROM PYTHON SOURCE LINES 49-58

.. code-block:: Python

    x = torch.randn((6, 7, 8))
    y = model(x)
    print(f"x.shape={x.shape}, y.shape={y.shape}")

    # Let's check it runs on another example.
    x2 = torch.randn((6, 8, 8))
    y2 = model(x2)
    print(f"x2.shape={x2.shape}, y2.shape={y2.shape}")





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    x.shape=torch.Size([6, 7, 8]), y.shape=torch.Size([6, 8, 8])
    x2.shape=torch.Size([6, 8, 8]), y2.shape=torch.Size([6, 8, 8])




.. GENERATED FROM PYTHON SOURCE LINES 59-63

Export
++++++

Let's defined the dynamic shapes and checks it exports.

.. GENERATED FROM PYTHON SOURCE LINES 63-70

.. code-block:: Python


    DYNAMIC = torch.export.Dim.DYNAMIC
    ep = torch.export.export(
        model, (x,), dynamic_shapes=({0: DYNAMIC, 1: DYNAMIC, 2: DYNAMIC},), strict=False
    )
    print(ep)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    ExportedProgram:
        class GraphModule(torch.nn.Module):
            def forward(self, x: "f32[s35, s16, s90]"):
                 # 
                sym_size_int_3: "Sym(s35)" = torch.ops.aten.sym_size.int(x, 0)
                sym_size_int_4: "Sym(s16)" = torch.ops.aten.sym_size.int(x, 1)
                sym_size_int_5: "Sym(s90)" = torch.ops.aten.sym_size.int(x, 2)
                add_1: "Sym(s16 + 3)" = 3 + sym_size_int_4
                floordiv_1: "Sym(((s16 + 3)//4))" = add_1 // 4;  add_1 = None
                mul_1: "Sym(4*(((s16 + 3)//4)))" = 4 * floordiv_1;  floordiv_1 = None
                le: "Sym(s16 <= 4*(((s16 + 3)//4)))" = sym_size_int_4 <= mul_1
                _assert_scalar_default = torch.ops.aten._assert_scalar.default(le, "Runtime assertion failed for expression s16 <= 4*(((s16 + 3)//4)) on node 'le'");  le = _assert_scalar_default = None
            
                 # File: /home/xadupre/github/experimental-experiment/_doc/recipes/plot_exporter_recipes_oe_dynpad.py:37 in forward, code: next_dim = ((dim + self.multiple - 1) // self.multiple) * self.multiple
                add: "Sym(s16 + 4)" = sym_size_int_4 + 4;  add = None
            
                 # File: /home/xadupre/github/experimental-experiment/_doc/recipes/plot_exporter_recipes_oe_dynpad.py:38 in forward, code: to_pad = next_dim - dim
                sub_1: "Sym(-s16 + 4*(((s16 + 3)//4)))" = mul_1 - sym_size_int_4;  mul_1 = sym_size_int_4 = None
            
                 # File: /home/xadupre/github/experimental-experiment/_doc/recipes/plot_exporter_recipes_oe_dynpad.py:39 in forward, code: pad = torch.zeros(
                zeros: "f32[s35, -s16 + 4*(((s16 + 3)//4)), s90]" = torch.ops.aten.zeros.default([sym_size_int_3, sub_1, sym_size_int_5], dtype = torch.float32, device = device(type='cpu'), pin_memory = False);  sym_size_int_3 = sub_1 = sym_size_int_5 = None
            
                 # File: /home/xadupre/github/experimental-experiment/_doc/recipes/plot_exporter_recipes_oe_dynpad.py:42 in forward, code: return torch.cat([x, pad], dim=self.dim_to_pad)
                cat: "f32[s35, 4*(((s16 + 3)//4)), s90]" = torch.ops.aten.cat.default([x, zeros], 1);  x = zeros = None
                return (cat,)
            
    Graph signature: 
        # inputs
        x: USER_INPUT
    
        # outputs
        cat: USER_OUTPUT
    
    Range constraints: {s35: VR[2, int_oo], s16: VR[2, int_oo], s90: VR[2, int_oo]}





.. GENERATED FROM PYTHON SOURCE LINES 71-72

We can also inline the local function.

.. GENERATED FROM PYTHON SOURCE LINES 72-77

.. code-block:: Python


    ep = torch.onnx.export(
        model, (x,), dynamic_shapes=({0: "batch", 1: "seq_len", 2: "num_frames"},), dynamo=True
    )





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    [torch.onnx] Obtain model graph for `PadToMultiple()` with `torch.export.export(..., strict=False)`...
    [torch.onnx] Obtain model graph for `PadToMultiple()` with `torch.export.export(..., strict=False)`... ✅
    [torch.onnx] Run decomposition...
    [torch.onnx] Run decomposition... ✅
    [torch.onnx] Translate the graph into ONNX...
    [torch.onnx] Translate the graph into ONNX... ✅
    Applied 2 of general pattern rewrite rules.




.. GENERATED FROM PYTHON SOURCE LINES 78-79

Let's save it.

.. GENERATED FROM PYTHON SOURCE LINES 79-81

.. code-block:: Python

    ep.save("plot_exporter_recipes_oe_dynpad.onnx")








.. GENERATED FROM PYTHON SOURCE LINES 82-86

Validation
++++++++++

Let's validate the exported model a set of inputs.

.. GENERATED FROM PYTHON SOURCE LINES 86-100

.. code-block:: Python

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





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    diff with shape=torch.Size([6, 8, 8]) -> torch.Size([6, 8, 8]): discrepancies=0.0
    diff with shape=torch.Size([6, 7, 8]) -> torch.Size([6, 8, 8]): discrepancies=0.0
    diff with shape=torch.Size([5, 8, 17]) -> torch.Size([5, 8, 17]): discrepancies=0.0
    diff with shape=torch.Size([1, 24, 4]) -> torch.Size([1, 24, 4]): discrepancies=0.0
    diff with shape=torch.Size([3, 9, 11]) -> torch.Size([3, 12, 11]): discrepancies=0.0




.. GENERATED FROM PYTHON SOURCE LINES 101-102

And visually.

.. GENERATED FROM PYTHON SOURCE LINES 102-104

.. code-block:: Python


    plot_dot(ep.model_proto)



.. image-sg:: /auto_recipes/images/sphx_glr_plot_exporter_recipes_oe_dynpad_001.png
   :alt: plot exporter recipes oe dynpad
   :srcset: /auto_recipes/images/sphx_glr_plot_exporter_recipes_oe_dynpad_001.png
   :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 1.242 seconds)


.. _sphx_glr_download_auto_recipes_plot_exporter_recipes_oe_dynpad.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_exporter_recipes_oe_dynpad.ipynb <plot_exporter_recipes_oe_dynpad.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_exporter_recipes_oe_dynpad.py <plot_exporter_recipes_oe_dynpad.py>`

    .. container:: sphx-glr-download sphx-glr-download-zip

      :download:`Download zipped: plot_exporter_recipes_oe_dynpad.zip <plot_exporter_recipes_oe_dynpad.zip>`


.. include:: plot_exporter_recipes_oe_dynpad.recommendations


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
