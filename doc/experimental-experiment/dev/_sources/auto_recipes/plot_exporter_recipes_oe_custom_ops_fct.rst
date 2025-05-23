
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_recipes/plot_exporter_recipes_oe_custom_ops_fct.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_recipes_plot_exporter_recipes_oe_custom_ops_fct.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_recipes_plot_exporter_recipes_oe_custom_ops_fct.py:


.. _l-plot-exporter-recipes-onnx-exporter-custom-ops-fct:

torch.onnx.export and a custom operator registered with a function
==================================================================

This example shows how to convert a custom operator, inspired from
`Python Custom Operators
<https://pytorch.org/tutorials/advanced/python_custom_ops.html#python-custom-ops-tutorial>`_.

A model with a custom ops
+++++++++++++++++++++++++

.. GENERATED FROM PYTHON SOURCE LINES 14-21

.. code-block:: Python


    import numpy as np
    from onnx.printer import to_text
    import onnxscript
    import torch









.. GENERATED FROM PYTHON SOURCE LINES 22-23

We define a model with a custom operator.

.. GENERATED FROM PYTHON SOURCE LINES 23-38

.. code-block:: Python



    def numpy_sin(x: torch.Tensor) -> torch.Tensor:
        assert x.device.type == "cpu"
        x_np = x.numpy()
        return torch.from_numpy(np.sin(x_np))


    class ModuleWithACustomOperator(torch.nn.Module):
        def forward(self, x):
            return numpy_sin(x)


    model = ModuleWithACustomOperator()








.. GENERATED FROM PYTHON SOURCE LINES 39-40

Let's check it runs.

.. GENERATED FROM PYTHON SOURCE LINES 40-43

.. code-block:: Python

    x = torch.randn(1, 3)
    model(x)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none


    tensor([[-0.1049, -0.2033,  0.8686]])



.. GENERATED FROM PYTHON SOURCE LINES 44-45

As expected, it does not export.

.. GENERATED FROM PYTHON SOURCE LINES 45-51

.. code-block:: Python

    try:
        torch.export.export(model, (x,))
        raise AssertionError("This export should failed unless pytorch now supports this model.")
    except Exception as e:
        print(e)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    .numpy() is not supported for tensor subclasses.




.. GENERATED FROM PYTHON SOURCE LINES 52-53

The exporter fails with the same eror as it expects torch.export.export to work.

.. GENERATED FROM PYTHON SOURCE LINES 53-60

.. code-block:: Python


    try:
        torch.onnx.export(model, (x,), dynamo=True)
    except Exception as e:
        print(e)






.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    [torch.onnx] Obtain model graph for `ModuleWithACustomOperator()` with `torch.export.export(..., strict=False)`...
    [torch.onnx] Obtain model graph for `ModuleWithACustomOperator()` with `torch.export.export(..., strict=False)`... ❌
    [torch.onnx] Obtain model graph for `ModuleWithACustomOperator()` with `torch.export.export(..., strict=True)`...
    [torch.onnx] Obtain model graph for `ModuleWithACustomOperator()` with `torch.export.export(..., strict=True)`... ✅
    [torch.onnx] Run decomposition...
    [torch.onnx] Run decomposition... ✅
    [torch.onnx] Translate the graph into ONNX...
    [torch.onnx] Translate the graph into ONNX... ✅




.. GENERATED FROM PYTHON SOURCE LINES 61-69

Registration
++++++++++++

The exporter how to convert the new exporter into ONNX.
This must be defined. The first piece is to tell the exporter
that the shape of the output is the same as x.
input names must be the same.
We also need to rewrite the module to be able to use it.

.. GENERATED FROM PYTHON SOURCE LINES 69-80

.. code-block:: Python



    def register(fct, fct_shape, namespace, fname):
        schema_str = torch.library.infer_schema(fct, mutates_args=())
        custom_def = torch.library.CustomOpDef(namespace, fname, schema_str, fct)
        custom_def.register_kernel("cpu")(fct)
        custom_def._abstract_fn = fct_shape


    register(numpy_sin, lambda x: torch.empty_like(x), "mylib", "numpy_sin")








.. GENERATED FROM PYTHON SOURCE LINES 81-82

We also need to rewrite the module to be able to use it.

.. GENERATED FROM PYTHON SOURCE LINES 82-91

.. code-block:: Python



    class ModuleWithACustomOperator(torch.nn.Module):
        def forward(self, x):
            return torch.ops.mylib.numpy_sin(x)


    model = ModuleWithACustomOperator()








.. GENERATED FROM PYTHON SOURCE LINES 92-93

Let's check it runs again.

.. GENERATED FROM PYTHON SOURCE LINES 93-95

.. code-block:: Python

    model(x)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none


    tensor([[-0.1049, -0.2033,  0.8686]])



.. GENERATED FROM PYTHON SOURCE LINES 96-97

Let's see what the fx graph looks like.

.. GENERATED FROM PYTHON SOURCE LINES 97-100

.. code-block:: Python


    print(torch.export.export(model, (x,)).graph)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    graph():
        %x : [num_users=1] = placeholder[target=x]
        %numpy_sin : [num_users=1] = call_function[target=torch.ops.mylib.numpy_sin.default](args = (%x,), kwargs = {})
        return (numpy_sin,)




.. GENERATED FROM PYTHON SOURCE LINES 101-102

Next is the conversion to onnx.

.. GENERATED FROM PYTHON SOURCE LINES 102-111

.. code-block:: Python


    op = onnxscript.opset18


    @onnxscript.script()
    def numpy_sin_to_onnx(x) -> onnxscript.onnx_types.TensorType:
        return op.Sin(x)









.. GENERATED FROM PYTHON SOURCE LINES 112-113

And we convert again.

.. GENERATED FROM PYTHON SOURCE LINES 113-122

.. code-block:: Python


    ep = torch.onnx.export(
        model,
        (x,),
        custom_translation_table={torch.ops.mylib.numpy_sin.default: numpy_sin_to_onnx},
        dynamo=True,
    )

    print(to_text(ep.model_proto))




.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    [torch.onnx] Obtain model graph for `ModuleWithACustomOperator()` with `torch.export.export(..., strict=False)`...
    [torch.onnx] Obtain model graph for `ModuleWithACustomOperator()` with `torch.export.export(..., strict=False)`... ✅
    [torch.onnx] Run decomposition...
    [torch.onnx] Run decomposition... ✅
    [torch.onnx] Translate the graph into ONNX...
    [torch.onnx] Translate the graph into ONNX... ✅
    <
       ir_version: 10,
       opset_import: ["" : 18],
       producer_name: "pytorch",
       producer_version: "2.8.0.dev20250519+cu126"
    >
    main_graph (float[1,3] x) => (float[1,3] numpy_sin) {
       [n0] numpy_sin = Sin (x)
    }





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 1.434 seconds)


.. _sphx_glr_download_auto_recipes_plot_exporter_recipes_oe_custom_ops_fct.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_exporter_recipes_oe_custom_ops_fct.ipynb <plot_exporter_recipes_oe_custom_ops_fct.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_exporter_recipes_oe_custom_ops_fct.py <plot_exporter_recipes_oe_custom_ops_fct.py>`

    .. container:: sphx-glr-download sphx-glr-download-zip

      :download:`Download zipped: plot_exporter_recipes_oe_custom_ops_fct.zip <plot_exporter_recipes_oe_custom_ops_fct.zip>`


.. include:: plot_exporter_recipes_oe_custom_ops_fct.recommendations


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
