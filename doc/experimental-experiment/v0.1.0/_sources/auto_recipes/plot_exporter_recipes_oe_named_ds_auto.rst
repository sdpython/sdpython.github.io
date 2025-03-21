
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_recipes/plot_exporter_recipes_oe_named_ds_auto.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_recipes_plot_exporter_recipes_oe_named_ds_auto.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_recipes_plot_exporter_recipes_oe_named_ds_auto.py:


.. _l-plot-exporter-recipes-onnx-exporter-modules:

torch.onnx.export: Rename Dynamic Shapes
========================================

Example given in :ref:`l-plot-exporter-dynamic_shapes` can only be exported
with dynamic shapes using ``torch.export.Dim.AUTO``. As a result, the exported
onnx models have dynamic dimensions with unpredictable names.

Model with unpredictable names for dynamic shapes
+++++++++++++++++++++++++++++++++++++++++++++++++

.. GENERATED FROM PYTHON SOURCE LINES 14-30

.. code-block:: Python


    import torch
    from experimental_experiment.helpers import pretty_onnx


    class Model(torch.nn.Module):
        def forward(self, x, y, z):
            return torch.cat((x, y), axis=1) + z[:, ::2]


    model = Model()
    x = torch.randn(2, 3)
    y = torch.randn(2, 5)
    z = torch.randn(2, 16)
    model(x, y, z)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none


    tensor([[ 1.7332,  2.5631,  3.2852, -1.5264,  0.3001,  0.5521, -0.3925, -0.0207],
            [-0.9006, -0.4954,  0.3012, -2.6659,  0.8444,  2.6513,  1.4366, -0.0708]])



.. GENERATED FROM PYTHON SOURCE LINES 31-32

Let's export it.

.. GENERATED FROM PYTHON SOURCE LINES 32-40

.. code-block:: Python


    AUTO = torch.export.Dim.AUTO
    ep = torch.export.export(
        model,
        (x, y, z),
        dynamic_shapes=({0: AUTO, 1: AUTO}, {0: AUTO, 1: AUTO}, {0: AUTO, 1: AUTO}),
    )








.. GENERATED FROM PYTHON SOURCE LINES 41-42

Let's convert it into ONNX.

.. GENERATED FROM PYTHON SOURCE LINES 42-50

.. code-block:: Python


    onx = torch.onnx.export(ep).model_proto

    for inp in onx.graph.input:
        print(f" input: {pretty_onnx(inp)}")
    for out in onx.graph.output:
        print(f"output: {pretty_onnx(out)}")





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    [torch.onnx] Run decomposition...
    [torch.onnx] Run decomposition... ✅
    [torch.onnx] Translate the graph into ONNX...
    [torch.onnx] Translate the graph into ONNX... ✅
    Applied 1 of general pattern rewrite rules.
     input: EXTERNAL[s0,s1] x
     input: EXTERNAL[s0,s3] y
     input: EXTERNAL[s0,s5] z
    output: EXTERNAL[s0,s1 + s3] add_11




.. GENERATED FROM PYTHON SOURCE LINES 51-57

Rename the dynamic shapes
+++++++++++++++++++++++++

We just need to give the onnx exporter the same information
:func:`torch.export.export` was given but we replace ``AUTO``
by the name this dimension should have.

.. GENERATED FROM PYTHON SOURCE LINES 57-72

.. code-block:: Python


    onx = torch.onnx.export(
        ep,
        dynamic_shapes=(
            {0: "batch", 1: "dx"},
            {0: "batch", 1: "dy"},
            {0: "batch", 1: "dx+dy"},
        ),
    ).model_proto

    for inp in onx.graph.input:
        print(f" input: {pretty_onnx(inp)}")
    for out in onx.graph.output:
        print(f"output: {pretty_onnx(out)}")





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    [torch.onnx] Run decomposition...
    [torch.onnx] Run decomposition... ✅
    [torch.onnx] Translate the graph into ONNX...
    [torch.onnx] Translate the graph into ONNX... ✅
    /home/xadupre/vv/this312/lib/python3.12/site-packages/torch/onnx/_internal/exporter/_dynamic_shapes.py:253: UserWarning: # The axis name: batch will not be used, since it shares the same shape constraints with another axis: batch.
      warnings.warn(
    Applied 1 of general pattern rewrite rules.
     input: EXTERNAL[batch,dx] x
     input: EXTERNAL[batch,dy] y
     input: EXTERNAL[batch,dx+dy] z
    output: EXTERNAL[batch,dx + dy] add_11




.. GENERATED FROM PYTHON SOURCE LINES 73-75

A model with an unknown output shape
++++++++++++++++++++++++++++++++++++

.. GENERATED FROM PYTHON SOURCE LINES 75-86

.. code-block:: Python



    class UnknownOutputModel(torch.nn.Module):
        def forward(self, x):
            return torch.nonzero(x)


    model = UnknownOutputModel()
    x = torch.randint(0, 2, (10, 2))
    model(x)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none


    tensor([[1, 0],
            [2, 0],
            [2, 1],
            [4, 0],
            [5, 1],
            [6, 1],
            [7, 0],
            [7, 1],
            [8, 0],
            [8, 1],
            [9, 0],
            [9, 1]])



.. GENERATED FROM PYTHON SOURCE LINES 87-88

Let's export it.

.. GENERATED FROM PYTHON SOURCE LINES 88-96

.. code-block:: Python


    ep = torch.export.export(
        model,
        (x,),
        dynamic_shapes=({0: torch.export.Dim("batch"), 1: AUTO},),
    )
    print(ep)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    ExportedProgram:
        class GraphModule(torch.nn.Module):
            def forward(self, x: "i64[s0, s1]"):
                 # File: /home/xadupre/github/experimental-experiment/_doc/recipes/plot_exporter_recipes_oe_named_ds_auto.py:79 in forward, code: return torch.nonzero(x)
                nonzero: "i64[u0, 2]" = torch.ops.aten.nonzero.default(x);  x = None
            
                 # 
                sym_size_int_3: "Sym(u0)" = torch.ops.aten.sym_size.int(nonzero, 0)
                sym_constrain_range_for_size_default = torch.ops.aten.sym_constrain_range_for_size.default(sym_size_int_3);  sym_constrain_range_for_size_default = None
            
                 # File: /home/xadupre/github/experimental-experiment/_doc/recipes/plot_exporter_recipes_oe_named_ds_auto.py:79 in forward, code: return torch.nonzero(x)
                ge_1: "Sym(u0 >= 0)" = sym_size_int_3 >= 0;  sym_size_int_3 = None
                _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u0 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_default = None
                return (nonzero,)
            
    Graph signature: ExportGraphSignature(input_specs=[InputSpec(kind=<InputKind.USER_INPUT: 1>, arg=TensorArgument(name='x'), target=None, persistent=None)], output_specs=[OutputSpec(kind=<OutputKind.USER_OUTPUT: 1>, arg=TensorArgument(name='nonzero'), target=None)])
    Range constraints: {u0: VR[0, 9223372036854775806], u1: VR[0, 9223372036854775806], s0: VR[0, int_oo], s1: VR[2, int_oo]}





.. GENERATED FROM PYTHON SOURCE LINES 97-98

Let's export it into ONNX.

.. GENERATED FROM PYTHON SOURCE LINES 98-106

.. code-block:: Python


    onx = torch.onnx.export(ep, dynamic_shapes=({0: "batch", 1: "dx"},), dynamo=True).model_proto

    for inp in onx.graph.input:
        print(f" input: {pretty_onnx(inp)}")
    for out in onx.graph.output:
        print(f"output: {pretty_onnx(out)}")





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    [torch.onnx] Run decomposition...
    [torch.onnx] Run decomposition... ✅
    [torch.onnx] Translate the graph into ONNX...
    [torch.onnx] Translate the graph into ONNX... ✅
     input: INT64[batch,dx] x
    output: INT64[u0,2] nonzero




.. GENERATED FROM PYTHON SOURCE LINES 107-113

The exporter has detected a dimension could not be infered
from the input shape somewhere in the graph and introduced a
new dimension name.
Let's rename it as well. Let's also change the output name
because the functionality may not be implemented yet when
the output dynamic shapes are given as a tuple.

.. GENERATED FROM PYTHON SOURCE LINES 113-134

.. code-block:: Python


    try:
        onx = torch.onnx.export(
            ep,
            dynamic_shapes=({0: "batch", 1: "dx"},),
            output_dynamic_shapes={"zeros": {0: "num_zeros"}},
            output_names=["zeros"],
            dynamo=True,
        ).model_proto
        raise AssertionError(
            "able to rename output dynamic dimensions, please update the tutorial"
        )
    except (TypeError, torch.onnx._internal.exporter._errors.ConversionError) as e:
        print(f"unable to rename output dynamic dimensions due to {e}")
        onx = None

    if onx is not None:
        for inp in onx.graph.input:
            print(f" input: {pretty_onnx(inp)}")
        for out in onx.graph.output:
            print(f"output: {pretty_onnx(out)}")




.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    unable to rename output dynamic dimensions due to export() got an unexpected keyword argument 'output_dynamic_shapes'





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 3.220 seconds)


.. _sphx_glr_download_auto_recipes_plot_exporter_recipes_oe_named_ds_auto.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_exporter_recipes_oe_named_ds_auto.ipynb <plot_exporter_recipes_oe_named_ds_auto.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_exporter_recipes_oe_named_ds_auto.py <plot_exporter_recipes_oe_named_ds_auto.py>`

    .. container:: sphx-glr-download sphx-glr-download-zip

      :download:`Download zipped: plot_exporter_recipes_oe_named_ds_auto.zip <plot_exporter_recipes_oe_named_ds_auto.zip>`


.. include:: plot_exporter_recipes_oe_named_ds_auto.recommendations


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
