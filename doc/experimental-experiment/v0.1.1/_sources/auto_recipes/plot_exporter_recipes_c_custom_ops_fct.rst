
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_recipes/plot_exporter_recipes_c_custom_ops_fct.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_recipes_plot_exporter_recipes_c_custom_ops_fct.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_recipes_plot_exporter_recipes_c_custom_ops_fct.py:


.. _l-plot-exporter-recipes-custom-custom-ops-fct:

to_onnx and a custom operator registered with a function
========================================================

This example shows how to convert a custom operator, inspired from
`Python Custom Operators
<https://pytorch.org/tutorials/advanced/python_custom_ops.html#python-custom-ops-tutorial>`_.

A model with a custom ops
+++++++++++++++++++++++++

.. GENERATED FROM PYTHON SOURCE LINES 14-24

.. code-block:: Python


    from typing import Any, Dict, List
    import numpy as np
    import torch
    from onnx_array_api.plotting.graphviz_helper import plot_dot
    from experimental_experiment.xbuilder import GraphBuilder
    from experimental_experiment.helpers import pretty_onnx
    from experimental_experiment.torch_interpreter import to_onnx, Dispatcher









.. GENERATED FROM PYTHON SOURCE LINES 25-26

We define a model with a custom operator.

.. GENERATED FROM PYTHON SOURCE LINES 26-41

.. code-block:: Python



    def numpy_sin(x: torch.Tensor) -> torch.Tensor:
        assert x.device.type == "cpu"
        x_np = x.numpy()
        return torch.from_numpy(np.sin(x_np))


    class ModuleWithACustomOperator(torch.nn.Module):
        def forward(self, x):
            return numpy_sin(x)


    model = ModuleWithACustomOperator()








.. GENERATED FROM PYTHON SOURCE LINES 42-43

Let's check it runs.

.. GENERATED FROM PYTHON SOURCE LINES 43-46

.. code-block:: Python

    x = torch.randn(1, 3)
    model(x)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none


    tensor([[-0.7577, -0.2187,  0.5606]])



.. GENERATED FROM PYTHON SOURCE LINES 47-48

As expected, it does not export.

.. GENERATED FROM PYTHON SOURCE LINES 48-54

.. code-block:: Python

    try:
        torch.export.export(model, (x,))
        raise AssertionError("This export should failed unless pytorch now supports this model.")
    except Exception as e:
        print(e)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    .numpy() is not supported for tensor subclasses.




.. GENERATED FROM PYTHON SOURCE LINES 55-56

The exporter fails with the same eror as it expects torch.export.export to work.

.. GENERATED FROM PYTHON SOURCE LINES 56-63

.. code-block:: Python


    try:
        to_onnx(model, (x,))
    except Exception as e:
        print(e)






.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    .numpy() is not supported for tensor subclasses.




.. GENERATED FROM PYTHON SOURCE LINES 64-71

Registration
++++++++++++

The exporter how to convert the new exporter into ONNX.
This must be defined. The first piece is to tell the exporter
that the shape of the output is the same as x.
input names must be the same.

.. GENERATED FROM PYTHON SOURCE LINES 71-90

.. code-block:: Python



    def register(fct, fct_shape, namespace, fname):
        schema_str = torch.library.infer_schema(fct, mutates_args=())
        custom_def = torch.library.CustomOpDef(namespace, fname, schema_str, fct)
        custom_def.register_kernel("cpu")(fct)
        custom_def._abstract_fn = fct_shape


    register(numpy_sin, lambda x: torch.empty_like(x), "mylib", "numpy_sin")


    class ModuleWithACustomOperator(torch.nn.Module):
        def forward(self, x):
            return torch.ops.mylib.numpy_sin(x)


    model = ModuleWithACustomOperator()








.. GENERATED FROM PYTHON SOURCE LINES 91-92

Let's check it runs again.

.. GENERATED FROM PYTHON SOURCE LINES 92-94

.. code-block:: Python

    model(x)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none


    tensor([[-0.7577, -0.2187,  0.5606]])



.. GENERATED FROM PYTHON SOURCE LINES 95-96

Let's see what the fx graph looks like.

.. GENERATED FROM PYTHON SOURCE LINES 96-99

.. code-block:: Python


    print(torch.export.export(model, (x,)).graph)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    graph():
        %x : [num_users=1] = placeholder[target=x]
        %numpy_sin : [num_users=1] = call_function[target=torch.ops.mylib.numpy_sin.default](args = (%x,), kwargs = {})
        return (numpy_sin,)




.. GENERATED FROM PYTHON SOURCE LINES 100-101

Next is the conversion to onnx.

.. GENERATED FROM PYTHON SOURCE LINES 101-115

.. code-block:: Python

    T = str  # a tensor name


    def numpy_sin_to_onnx(
        g: GraphBuilder,
        sts: Dict[str, Any],
        outputs: List[str],
        x: T,
        name: str = "mylib.numpy_sin",
    ) -> T:
        # name= ... lets the user know when the node comes from
        return g.op.Sin(x, name=name, outputs=outputs)









.. GENERATED FROM PYTHON SOURCE LINES 116-117

We create a :class:`Dispatcher <experimental_experiment.torch_interpreter.Dispatcher>`.

.. GENERATED FROM PYTHON SOURCE LINES 117-120

.. code-block:: Python


    dispatcher = Dispatcher({"mylib::numpy_sin": numpy_sin_to_onnx})








.. GENERATED FROM PYTHON SOURCE LINES 121-122

And we convert again.

.. GENERATED FROM PYTHON SOURCE LINES 122-126

.. code-block:: Python


    onx = to_onnx(model, (x,), dispatcher=dispatcher, optimize=False)
    print(pretty_onnx(onx))





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    opset: domain='' version=18
    input: name='x' type=dtype('float32') shape=[1, 3]
    Sin(x) -> numpy_sin
      Identity(numpy_sin) -> output_0
    output: name='output_0' type=dtype('float32') shape=[1, 3]




.. GENERATED FROM PYTHON SOURCE LINES 127-128

And we convert again with optimization this time.

.. GENERATED FROM PYTHON SOURCE LINES 128-132

.. code-block:: Python


    onx = to_onnx(model, (x,), dispatcher=dispatcher, optimize=True)
    print(pretty_onnx(onx))





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    opset: domain='' version=18
    input: name='x' type=dtype('float32') shape=[1, 3]
    Sin(x) -> output_0
    output: name='output_0' type=dtype('float32') shape=[1, 3]




.. GENERATED FROM PYTHON SOURCE LINES 133-135

Let's make sure the node was produce was the user defined converter for numpy_sin.
The name should be 'mylib.numpy_sin'.

.. GENERATED FROM PYTHON SOURCE LINES 135-138

.. code-block:: Python


    print(onx.graph.node[0])





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    input: "x"
    output: "output_0"
    name: "mylib.numpy_sin"
    op_type: "Sin"
    domain: ""





.. GENERATED FROM PYTHON SOURCE LINES 139-140

And visually.

.. GENERATED FROM PYTHON SOURCE LINES 140-142

.. code-block:: Python


    plot_dot(onx)



.. image-sg:: /auto_recipes/images/sphx_glr_plot_exporter_recipes_c_custom_ops_fct_001.png
   :alt: plot exporter recipes c custom ops fct
   :srcset: /auto_recipes/images/sphx_glr_plot_exporter_recipes_c_custom_ops_fct_001.png
   :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 0.228 seconds)


.. _sphx_glr_download_auto_recipes_plot_exporter_recipes_c_custom_ops_fct.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_exporter_recipes_c_custom_ops_fct.ipynb <plot_exporter_recipes_c_custom_ops_fct.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_exporter_recipes_c_custom_ops_fct.py <plot_exporter_recipes_c_custom_ops_fct.py>`

    .. container:: sphx-glr-download sphx-glr-download-zip

      :download:`Download zipped: plot_exporter_recipes_c_custom_ops_fct.zip <plot_exporter_recipes_c_custom_ops_fct.zip>`


.. include:: plot_exporter_recipes_c_custom_ops_fct.recommendations


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
