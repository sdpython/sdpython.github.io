
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_examples/plot_failing_onnxruntime_evaluator.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_examples_plot_failing_onnxruntime_evaluator.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_plot_failing_onnxruntime_evaluator.py:


.. _l-plot-failing-onnxruntime-evaluator:

Intermediate results with onnxruntime
=====================================

Example :ref:`l-plot-failing-reference-evaluator` demonstrated
how to run a python runtime on a model but it may very slow sometimes
and it could show some discrepancies if the only provider is not CPU.
Let's use :class:`OnnxruntimeEvaluator <onnx_diagnostic.reference.OnnxruntimeEvaluator>`.
It splits the model into node and runs them independently until it succeeds
or fails. This class converts every node into model based on the types
discovered during the execution. It relies on :class:`InferenceSessionForTorch
<onnx_diagnostic.helpers.ort_session.InferenceSessionForTorch>` or
:class:`InferenceSessionForNumpy
<onnx_diagnostic.helpers.ort_session.InferenceSessionForNumpy>`
for the execution. This example uses torch tensor and
bfloat16.

A failing model
+++++++++++++++

The issue here is a an operator ``Cast`` trying to convert a result
into a non-existing type.

.. GENERATED FROM PYTHON SOURCE LINES 26-59

.. code-block:: Python


    import onnx
    import onnx.helper as oh
    import torch
    import onnxruntime
    from onnx_diagnostic import doc
    from onnx_diagnostic.ext_test_case import has_cuda
    from onnx_diagnostic.helpers.onnx_helper import from_array_extended
    from onnx_diagnostic.reference import OnnxruntimeEvaluator

    TBFLOAT16 = onnx.TensorProto.BFLOAT16

    model = oh.make_model(
        oh.make_graph(
            [
                oh.make_node("Mul", ["X", "Y"], ["xy"], name="n0"),
                oh.make_node("Sigmoid", ["xy"], ["sy"], name="n1"),
                oh.make_node("Add", ["sy", "one"], ["C"], name="n2"),
                oh.make_node("Cast", ["C"], ["X999"], to=999, name="failing"),
                oh.make_node("CastLike", ["X999", "Y"], ["Z"], name="n4"),
            ],
            "-nd-",
            [
                oh.make_tensor_value_info("X", TBFLOAT16, ["a", "b", "c"]),
                oh.make_tensor_value_info("Y", TBFLOAT16, ["a", "b", "c"]),
            ],
            [oh.make_tensor_value_info("Z", TBFLOAT16, ["a", "b", "c"])],
            [from_array_extended(torch.tensor([1], dtype=torch.bfloat16), name="one")],
        ),
        opset_imports=[oh.make_opsetid("", 18)],
        ir_version=9,
    )








.. GENERATED FROM PYTHON SOURCE LINES 60-61

We check it is failing.

.. GENERATED FROM PYTHON SOURCE LINES 61-68

.. code-block:: Python


    try:
        onnxruntime.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    except onnxruntime.capi.onnxruntime_pybind11_state.Fail as e:
        print(e)






.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    [ONNXRuntimeError] : 1 : FAIL : Node (failing) Op (Cast) [TypeInferenceError] Attribute to does not specify a valid type in .




.. GENERATED FROM PYTHON SOURCE LINES 69-76

OnnxruntimeEvaluator
++++++++++++++++++++++++++

This class extends :class:`onnx.reference.ReferenceEvaluator`
with operators outside the standard but defined by :epkg:`onnxruntime`.
`verbose=10` tells the class to print as much as possible,
`verbose=0` prints nothing. Intermediate values for more or less verbosity.

.. GENERATED FROM PYTHON SOURCE LINES 76-87

.. code-block:: Python


    ref = OnnxruntimeEvaluator(model, verbose=10)
    feeds = dict(
        X=torch.rand((3, 4), dtype=torch.bfloat16), Y=torch.rand((3, 4), dtype=torch.bfloat16)
    )
    try:
        ref.run(None, feeds)
    except Exception as e:
        print("ERROR", type(e), e)






.. rst-class:: sphx-glr-script-out

 .. code-block:: none

     +C one: A:bfloat16:(1,):[1.0]
     +I X: T:D-1:torch.bfloat16:torch.Size([3, 4]):0.5625,0.46875,0.89453125,0.18359375,0.9375,0.04296875,0.71875,0.10546875,0.3515625,0.703125...
     +I Y: T:D-1:torch.bfloat16:torch.Size([3, 4]):0.08203125,0.01953125,0.98046875,0.828125,0.15625,0.33984375,0.2890625,0.16796875,0.2109375,0.20703125...
    Mul(X, Y) -> xy
    ERROR <class 'onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented'> [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for Mul(14) node with name 'n0'




.. GENERATED FROM PYTHON SOURCE LINES 88-90

:epkg:`onnxruntime` may not support bfloat16 on CPU.
See :epkg:`onnxruntime kernels`.

.. GENERATED FROM PYTHON SOURCE LINES 90-101

.. code-block:: Python


    if has_cuda():
        ref = OnnxruntimeEvaluator(model, providers="cuda", verbose=10)
        feeds = dict(
            X=torch.rand((3, 4), dtype=torch.bfloat16), Y=torch.rand((3, 4), dtype=torch.bfloat16)
        )
        try:
            ref.run(None, feeds)
        except Exception as e:
            print("ERROR", type(e), e)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

     +C one: A:bfloat16:(1,):[1.0]
     +I X: T:D-1:torch.bfloat16:torch.Size([3, 4]):0.3125,0.98046875,0.85546875,0.97265625,0.06640625,0.38671875,0.89453125,0.26953125,0.44921875,0.828125...
     +I Y: T:D-1:torch.bfloat16:torch.Size([3, 4]):0.5703125,0.98828125,0.50390625,0.64453125,0.3203125,0.7734375,0.2890625,0.125,0.6953125,0.39453125...
    Mul(X, Y) -> xy
     + xy: T:D-1:torch.bfloat16:torch.Size([3, 4]):0.177734375,0.96875,0.431640625,0.625,0.021240234375,0.298828125,0.2578125,0.03369140625,0.3125,0.326171875...
     - deletes: X - torch.bfloat16:torch.Size([3, 4])
    Sigmoid(xy) -> sy
     + sy: T:D-1:torch.bfloat16:torch.Size([3, 4]):0.54296875,0.7265625,0.60546875,0.65234375,0.50390625,0.57421875,0.5625,0.5078125,0.578125,0.58203125...
     - deletes: xy - torch.bfloat16:torch.Size([3, 4])
    Add(sy, one) -> C
     + C: A:bfloat16:(3, 4):1.546875,1.7265625,1.609375,1.65625,1.5,1.578125,1.5625,1.5078125,1.578125,1.578125...
     - deletes: sy - torch.bfloat16:torch.Size([3, 4])
     - deletes: one - bfloat16:(1,)
    Cast(C) -> X999
    ERROR <class 'RuntimeError'> Unable to create a session stored in '_debug_InferenceSession_last_failure.onnx'), providers=['CUDAExecutionProvider', 'CPUExecutionProvider']




.. GENERATED FROM PYTHON SOURCE LINES 102-108

We can see it run until it reaches `Cast` and stops.
The error message is not always obvious to interpret.
It gets improved every time from time to time.
This runtime is useful when it fails for a numerical reason.
It is possible to insert prints in the python code to print
more information or debug if needed.

.. GENERATED FROM PYTHON SOURCE LINES 108-110

.. code-block:: Python


    doc.plot_legend("onnxruntime\nrunning\nstep by step", "OnnxruntimeEvaluator", "lightgrey")



.. image-sg:: /auto_examples/images/sphx_glr_plot_failing_onnxruntime_evaluator_001.png
   :alt: plot failing onnxruntime evaluator
   :srcset: /auto_examples/images/sphx_glr_plot_failing_onnxruntime_evaluator_001.png
   :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 10.940 seconds)


.. _sphx_glr_download_auto_examples_plot_failing_onnxruntime_evaluator.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_failing_onnxruntime_evaluator.ipynb <plot_failing_onnxruntime_evaluator.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_failing_onnxruntime_evaluator.py <plot_failing_onnxruntime_evaluator.py>`

    .. container:: sphx-glr-download sphx-glr-download-zip

      :download:`Download zipped: plot_failing_onnxruntime_evaluator.zip <plot_failing_onnxruntime_evaluator.zip>`


.. include:: plot_failing_onnxruntime_evaluator.recommendations


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
