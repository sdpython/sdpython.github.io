
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_examples/plot_torch_custom_backend_101.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_examples_plot_torch_custom_backend_101.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_plot_torch_custom_backend_101.py:


.. _l-plot-custom-backend:

===============================
101: A custom backend for torch
===============================

This example leverages the examples introduced on this page
`Custom Backends <https://pytorch.org/docs/stable/torch.compiler_custom_backends.html>`_.
It uses backend :func:`experimental_experiment.torch_dynamo.onnx_custom_backend`
based on :epkg:`onnxruntime` and running on CPU or CUDA.
It could easily replaced by
:func:`experimental_experiment.torch_dynamo.onnx_debug_backend`.
This one based on the reference implemented from onnx
can show the intermediate results if needed. It is very slow.

A model
=======

.. GENERATED FROM PYTHON SOURCE LINES 20-53

.. code-block:: Python


    import copy
    from experimental_experiment.helpers import pretty_onnx
    from onnx_array_api.plotting.graphviz_helper import plot_dot
    import torch
    from torch._dynamo.backends.common import aot_autograd

    # from torch._functorch._aot_autograd.utils import make_boxed_func
    from experimental_experiment.torch_dynamo import (
        onnx_custom_backend,
        get_decomposition_table,
    )
    from experimental_experiment.torch_interpreter import ExportOptions


    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(10, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 1),
            )

        def forward(self, x):
            return self.layers(x)


    x = torch.randn(3, 10, dtype=torch.float32)

    mlp = MLP()
    print(mlp(x))





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    tensor([[0.0408],
            [0.1123],
            [0.1966]], grad_fn=<AddmmBackward0>)




.. GENERATED FROM PYTHON SOURCE LINES 54-62

A custom backend
================

This backend leverages :epkg:`onnxruntime`.
It is available through function
:func:`experimental_experiment.torch_dynamo.onnx_custom_backend`
and implemented by class :class:`OrtBackend
<experimental_experiment.torch_dynamo.fast_backend.OrtBackend>`.

.. GENERATED FROM PYTHON SOURCE LINES 62-72

.. code-block:: Python


    compiled_model = torch.compile(
        copy.deepcopy(mlp),
        backend=lambda *args, **kwargs: onnx_custom_backend(*args, target_opset=18, **kwargs),
        dynamic=False,
        fullgraph=True,
    )

    print(compiled_model(x))





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    tensor([[0.0408],
            [0.1123],
            [0.1966]])




.. GENERATED FROM PYTHON SOURCE LINES 73-85

Training
========

It can be used for training as well. The compilation may not
be working if the model is using function the converter does not know.
Maybe, there exist a way to decompose this new function into
existing functions. A recommended list is returned by
with function :func:`get_decomposition_table
<experimental_experiment.torch_dynamo.get_decomposition_table>`.
An existing list can be filtered out from some inefficient decompositions
with function :func:`filter_decomposition_table
<experimental_experiment.torch_dynamo.filter_decomposition_table>`.

.. GENERATED FROM PYTHON SOURCE LINES 85-105

.. code-block:: Python



    aot_compiler = aot_autograd(
        fw_compiler=lambda *args, **kwargs: onnx_custom_backend(
            *args,
            target_opset=18,
            export_options=ExportOptions(decomposition_table=get_decomposition_table()),
            **kwargs,
        ),
    )

    compiled_model = torch.compile(
        copy.deepcopy(mlp),
        backend=aot_compiler,
        fullgraph=True,
        dynamic=False,
    )

    print(compiled_model(x))





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    tensor([[0.0408],
            [0.1123],
            [0.1966]], grad_fn=<CompiledFunctionBackward>)




.. GENERATED FROM PYTHON SOURCE LINES 106-107

Let's see an iteration loop.

.. GENERATED FROM PYTHON SOURCE LINES 107-171

.. code-block:: Python


    from sklearn.datasets import load_diabetes


    class DiabetesDataset(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X = torch.from_numpy(X / 10).to(torch.float32)
            self.y = torch.from_numpy(y).to(torch.float32).reshape((-1, 1))

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            return self.X[i], self.y[i]


    def trained_model(max_iter=5, dynamic=False, storage=None):
        aot_compiler = aot_autograd(
            fw_compiler=lambda *args, **kwargs: onnx_custom_backend(
                *args, target_opset=18, storage=storage, **kwargs
            ),
            decompositions=get_decomposition_table(),
        )

        compiled_model = torch.compile(
            MLP(),
            backend=aot_compiler,
            fullgraph=True,
            dynamic=dynamic,
        )

        trainloader = torch.utils.data.DataLoader(
            DiabetesDataset(*load_diabetes(return_X_y=True)),
            batch_size=5,
            shuffle=True,
            num_workers=0,
        )

        loss_function = torch.nn.L1Loss()
        optimizer = torch.optim.Adam(compiled_model.parameters(), lr=1e-1)

        for epoch in range(0, max_iter):
            current_loss = 0.0

            for _, data in enumerate(trainloader, 0):
                X, y = data

                optimizer.zero_grad()
                p = compiled_model(X)
                loss = loss_function(p, y)
                loss.backward()

                optimizer.step()

                current_loss += loss.item()

            print(f"Loss after epoch {epoch+1}: {current_loss}")

        print("Training process has finished.")
        return compiled_model


    trained_model(3)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    /home/xadupre/vv/this312/lib/python3.12/site-packages/torch/_functorch/_aot_autograd/utils.py:130: UserWarning: Your compiler for AOTAutograd is returning a function that doesn't take boxed arguments. Please wrap it with functorch.compile.make_boxed_func or handle the boxed arguments yourself. See https://github.com/pytorch/pytorch/pull/83137#issuecomment-1211320670 for rationale.
      warnings.warn(
    /home/xadupre/vv/this312/lib/python3.12/site-packages/torch/_functorch/_aot_autograd/utils.py:130: UserWarning: Your compiler for AOTAutograd is returning a function that doesn't take boxed arguments. Please wrap it with functorch.compile.make_boxed_func or handle the boxed arguments yourself. See https://github.com/pytorch/pytorch/pull/83137#issuecomment-1211320670 for rationale.
      warnings.warn(
    Loss after epoch 1: 7593.021614074707
    Loss after epoch 2: 5520.917652130127
    Loss after epoch 3: 5328.389085769653
    Training process has finished.

    OptimizedModule(
      (_orig_mod): MLP(
        (layers): Sequential(
          (0): Linear(in_features=10, out_features=32, bias=True)
          (1): ReLU()
          (2): Linear(in_features=32, out_features=1, bias=True)
        )
      )
    )



.. GENERATED FROM PYTHON SOURCE LINES 172-177

What about the ONNX model?
==========================

The backend converts the model into ONNX then runs it with :epkg:`onnxruntime`.
Let's see what it looks like.

.. GENERATED FROM PYTHON SOURCE LINES 177-190

.. code-block:: Python


    storage = {}

    trained_model(3, storage=storage)

    print(f"{len(storage['instance'])} were created.")

    for i, inst in enumerate(storage["instance"][:2]):
        print()
        print(f"-- model {i} running on {inst['providers']}")
        print(pretty_onnx(inst["onnx"]))






.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    /home/xadupre/vv/this312/lib/python3.12/site-packages/torch/_functorch/_aot_autograd/utils.py:130: UserWarning: Your compiler for AOTAutograd is returning a function that doesn't take boxed arguments. Please wrap it with functorch.compile.make_boxed_func or handle the boxed arguments yourself. See https://github.com/pytorch/pytorch/pull/83137#issuecomment-1211320670 for rationale.
      warnings.warn(
    /home/xadupre/vv/this312/lib/python3.12/site-packages/torch/_functorch/_aot_autograd/utils.py:130: UserWarning: Your compiler for AOTAutograd is returning a function that doesn't take boxed arguments. Please wrap it with functorch.compile.make_boxed_func or handle the boxed arguments yourself. See https://github.com/pytorch/pytorch/pull/83137#issuecomment-1211320670 for rationale.
      warnings.warn(
    Loss after epoch 1: 7318.274580001831
    Loss after epoch 2: 5505.066953659058
    Loss after epoch 3: 5255.837236404419
    Training process has finished.
    4 were created.

    -- model 0 running on ['CPUExecutionProvider']
    opset: domain='' version=18
    input: name='input0' type=dtype('float32') shape=[32, 10]
    input: name='input1' type=dtype('float32') shape=[32]
    input: name='input2' type=dtype('float32') shape=[5, 10]
    input: name='input3' type=dtype('float32') shape=[1, 32]
    input: name='input4' type=dtype('float32') shape=[1]
    init: name='init7_s2_-1_1' type=int64 shape=(2,) -- array([-1,  1])   -- TransposeEqualReshapePattern.apply.new_shape
    Gemm(input2, input0, input1, transA=0, transB=1, alpha=1.00, beta=1.00) -> addmm
      Relu(addmm) -> output_2
    Reshape(input3, init7_s2_-1_1) -> output_3
      Gemm(output_2, output_3, input4, alpha=1.00, beta=1.00) -> output_0
    Identity(input2) -> output_1
    output: name='output_0' type=dtype('float32') shape=[5, 1]
    output: name='output_1' type=dtype('float32') shape=[5, 10]
    output: name='output_2' type=dtype('float32') shape=[5, 32]
    output: name='output_3' type=dtype('float32') shape=[32, 1]

    -- model 1 running on ['CPUExecutionProvider']
    opset: domain='' version=18
    input: name='input0' type=dtype('float32') shape=[5, 10]
    input: name='input1' type=dtype('float32') shape=[5, 32]
    input: name='input2' type=dtype('float32') shape=[32, 1]
    input: name='input3' type=dtype('float32') shape=[5, 1]
    init: name='init7_s1_0' type=int64 shape=(1,) -- array([0])           -- Opset.make_node.1/Shape##Opset.make_node.1/Shape
    init: name='init1_s1_' type=float32 shape=(1,) -- array([0.], dtype=float32)-- Opset.make_node.1/Small##Opset.make_node.1/Small
    init: name='init7_s2_1_-1' type=int64 shape=(2,) -- array([ 1, -1])   -- TransposeEqualReshapePattern.apply.new_shape##TransposeEqualReshapePattern.apply.new_shape
    Constant(value_float=0.0) -> output_NONE_2
    Reshape(input2, init7_s2_1_-1) -> t_2
      MatMul(input3, t_2) -> mm
    Reshape(input3, init7_s2_1_-1) -> t_3
      MatMul(t_3, input1) -> output_3
    ReduceSum(input3, init7_s1_0, keepdims=0) -> output_4
    LessOrEqual(input1, init1_s1_) -> _onx_lessorequal_detach_30
      Where(_onx_lessorequal_detach_30, init1_s1_, mm) -> threshold_backward
        Gemm(threshold_backward, input0, transA=1, transB=0) -> output_0
    ReduceSum(threshold_backward, init7_s1_0, keepdims=0) -> output_1
    output: name='output_0' type=dtype('float32') shape=[32, 10]
    output: name='output_1' type=dtype('float32') shape=[32]
    output: name='output_NONE_2' type=dtype('float32') shape=None
    output: name='output_3' type=dtype('float32') shape=[1, 32]
    output: name='output_4' type=dtype('float32') shape=[1]




.. GENERATED FROM PYTHON SOURCE LINES 191-192

The forward graph.

.. GENERATED FROM PYTHON SOURCE LINES 192-196

.. code-block:: Python


    plot_dot(storage["instance"][0]["onnx"])





.. image-sg:: /auto_examples/images/sphx_glr_plot_torch_custom_backend_101_001.png
   :alt: plot torch custom backend 101
   :srcset: /auto_examples/images/sphx_glr_plot_torch_custom_backend_101_001.png
   :class: sphx-glr-single-img





.. GENERATED FROM PYTHON SOURCE LINES 197-198

The backward graph.

.. GENERATED FROM PYTHON SOURCE LINES 198-202

.. code-block:: Python


    plot_dot(storage["instance"][1]["onnx"])





.. image-sg:: /auto_examples/images/sphx_glr_plot_torch_custom_backend_101_002.png
   :alt: plot torch custom backend 101
   :srcset: /auto_examples/images/sphx_glr_plot_torch_custom_backend_101_002.png
   :class: sphx-glr-single-img





.. GENERATED FROM PYTHON SOURCE LINES 203-209

What about dynamic shapes?
==========================

Any input or output having `_dim_` in its name is a dynamic dimension.
Any output having `_NONE_` in its name is replace by None.
It is needed by pytorch.

.. GENERATED FROM PYTHON SOURCE LINES 209-222

.. code-block:: Python


    storage = {}

    trained_model(3, storage=storage, dynamic=True)

    print(f"{len(storage['instance'])} were created.")

    for i, inst in enumerate(storage["instance"]):
        print()
        print(f"-- model {i} running on {inst['providers']}")
        print()
        print(pretty_onnx(inst["onnx"]))





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    /home/xadupre/vv/this312/lib/python3.12/site-packages/torch/_functorch/_aot_autograd/utils.py:130: UserWarning: Your compiler for AOTAutograd is returning a function that doesn't take boxed arguments. Please wrap it with functorch.compile.make_boxed_func or handle the boxed arguments yourself. See https://github.com/pytorch/pytorch/pull/83137#issuecomment-1211320670 for rationale.
      warnings.warn(
    Loss after epoch 1: 7330.134973526001
    Loss after epoch 2: 5513.183366775513
    Loss after epoch 3: 5139.226182937622
    Training process has finished.
    2 were created.

    -- model 0 running on ['CPUExecutionProvider']

    opset: domain='' version=18
    input: name='input0' type=dtype('float32') shape=[32, 10]
    input: name='input1' type=dtype('float32') shape=[32]
    input: name='input_dim_2' type=dtype('int64') shape=None
    input: name='input3' type=dtype('float32') shape=['s77', 10]
    input: name='input4' type=dtype('float32') shape=[1, 32]
    input: name='input5' type=dtype('float32') shape=[1]
    init: name='init7_s2_-1_1' type=int64 shape=(2,) -- array([-1,  1])   -- TransposeEqualReshapePattern.apply.new_shape
    Gemm(input3, input0, input1, transA=0, transB=1, alpha=1.00, beta=1.00) -> addmm
      Relu(addmm) -> output_2
    Reshape(input4, init7_s2_-1_1) -> output_3
      Gemm(output_2, output_3, input5, alpha=1.00, beta=1.00) -> output_0
    Identity(input3) -> output_1
    Identity(input_dim_2) -> output_dim_4
    output: name='output_0' type=dtype('float32') shape=['s77', 1]
    output: name='output_1' type=dtype('float32') shape=['s77', 10]
    output: name='output_2' type=dtype('float32') shape=['s77', 32]
    output: name='output_3' type=dtype('float32') shape=[32, 1]
    output: name='output_dim_4' type=dtype('int64') shape=None

    -- model 1 running on ['CPUExecutionProvider']

    opset: domain='' version=18
    input: name='input_dim_0' type=dtype('int64') shape=None
    input: name='input1' type=dtype('float32') shape=['s77', 10]
    input: name='input2' type=dtype('float32') shape=['s77', 32]
    input: name='input3' type=dtype('float32') shape=[32, 1]
    input: name='input4' type=dtype('float32') shape=['s77', 1]
    init: name='init7_s1_0' type=int64 shape=(1,) -- array([0])           -- Opset.make_node.1/Shape##Opset.make_node.1/Shape
    init: name='init1_s1_' type=float32 shape=(1,) -- array([0.], dtype=float32)-- Opset.make_node.1/Small##Opset.make_node.1/Small
    init: name='init7_s2_1_-1' type=int64 shape=(2,) -- array([ 1, -1])   -- TransposeEqualReshapePattern.apply.new_shape##TransposeEqualReshapePattern.apply.new_shape
    Constant(value_float=0.0) -> output_NONE_2
      Identity(output_NONE_2) -> output_NONE_3
    Reshape(input3, init7_s2_1_-1) -> t_2
      MatMul(input4, t_2) -> mm
    Reshape(input4, init7_s2_1_-1) -> t_3
      MatMul(t_3, input2) -> output_4
    ReduceSum(input4, init7_s1_0, keepdims=0) -> output_5
    LessOrEqual(input2, init1_s1_) -> _onx_lessorequal_detach_30
      Where(_onx_lessorequal_detach_30, init1_s1_, mm) -> threshold_backward
        Gemm(threshold_backward, input1, transA=1, transB=0) -> output_0
    ReduceSum(threshold_backward, init7_s1_0, keepdims=0) -> output_1
    output: name='output_0' type=dtype('float32') shape=[32, 10]
    output: name='output_1' type=dtype('float32') shape=[32]
    output: name='output_NONE_2' type=dtype('float32') shape=None
    output: name='output_NONE_3' type=dtype('float32') shape=None
    output: name='output_4' type=dtype('float32') shape=[1, 32]
    output: name='output_5' type=dtype('float32') shape=[1]




.. GENERATED FROM PYTHON SOURCE LINES 223-224

The forward graph.

.. GENERATED FROM PYTHON SOURCE LINES 224-228

.. code-block:: Python


    plot_dot(storage["instance"][0]["onnx"])





.. image-sg:: /auto_examples/images/sphx_glr_plot_torch_custom_backend_101_003.png
   :alt: plot torch custom backend 101
   :srcset: /auto_examples/images/sphx_glr_plot_torch_custom_backend_101_003.png
   :class: sphx-glr-single-img





.. GENERATED FROM PYTHON SOURCE LINES 229-230

The backward graph.

.. GENERATED FROM PYTHON SOURCE LINES 230-234

.. code-block:: Python


    plot_dot(storage["instance"][1]["onnx"])





.. image-sg:: /auto_examples/images/sphx_glr_plot_torch_custom_backend_101_004.png
   :alt: plot torch custom backend 101
   :srcset: /auto_examples/images/sphx_glr_plot_torch_custom_backend_101_004.png
   :class: sphx-glr-single-img





.. GENERATED FROM PYTHON SOURCE LINES 235-243

Pattern Optimizations
=====================

By default, once exported into onnx, a model is optimized by
looking for patterns. Each of them locally replaces a couple of
nodes to optimize the computation
(see :ref:`l-pattern-optimization-onnx` and
:ref:`l-pattern-optimization-ort`).


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 16.572 seconds)


.. _sphx_glr_download_auto_examples_plot_torch_custom_backend_101.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_torch_custom_backend_101.ipynb <plot_torch_custom_backend_101.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_torch_custom_backend_101.py <plot_torch_custom_backend_101.py>`

    .. container:: sphx-glr-download sphx-glr-download-zip

      :download:`Download zipped: plot_torch_custom_backend_101.zip <plot_torch_custom_backend_101.zip>`


.. include:: plot_torch_custom_backend_101.recommendations


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
