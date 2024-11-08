:orphan:

Example gallery
===============

A couple of examples to illustrate different implementation
of dot product (see also :epkg:`sphinx-gallery`).

Getting started
+++++++++++++++

pytorch nightly build should be installed, see
`Start Locally <https://pytorch.org/get-started/locally/>`_.

::

    git clone https://github.com/sdpython/experimental-experiment.git
    pip install onnxruntime-gpu pynvml
    pip install -r requirements-dev.txt    
    export PYTHONPATH=$PYTHONPATH:<this folder>

Compare torch exporters
+++++++++++++++++++++++

The script evaluates the memory peak, the computation time of the exporters.
It also compares the exported models when run through onnxruntime.
The full script takes around 20 minutes to complete. It stores on disk
all the graphs, the data used to draw them, and the models.

::

    python _doc/examples/plot_torch_export.py -s large



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script gathers a couple of examples based on onnxscript.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_onnxscript_102_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_onnxscript_102.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">102: Examples with onnxscript</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example shows how to convert a custom operator as defined in the tutorial Python Custom Operators.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_exporter_recipes_oe_custom_ops_inplace_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_exporter_recipes_oe_custom_ops_inplace.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">torch.onnx.export and a custom operator inplace</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Control flow cannot be exported with a change. The code of the model can be changed or patched to introduce function torch.cond.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_exporter_recipes_oe_cond_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_exporter_recipes_oe_cond.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">torch.onnx.export and a model with a test</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Control flow cannot be exported with a change. The code of the model can be changed or patched to introduce function torch.cond.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_exporter_recipes_c_cond_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_exporter_recipes_c_cond.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">to_onnx and a model with a test</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example shows how to convert a custom operator, inspired from Python Custom Operators.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_exporter_recipes_oe_custom_ops_fct_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_exporter_recipes_oe_custom_ops_fct.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">torch.onnx.export and a custom operator registered with a function</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example shows how to optimize a graph using pattern optimization. The graph was obtained by running a dummy llama model. It is the backward graph.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_optimize_101_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_optimize_101.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">101: Graph Optimization</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example shows how to convert a custom operator as defined in the tutorial Python Custom Operators.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_exporter_recipes_c_custom_ops_inplace_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_exporter_recipes_c_custom_ops_inplace.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">to_onnx and a custom operator inplace</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script demonstrates ExecuTorch on a very simple example, see also ExecuTorch Tutorial, ExecuTorch Runtime Python API Reference.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_executorch_102_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_executorch_102.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">102: First test with ExecuTorch</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example shows how to convert a custom operator, inspired from Python Custom Operators.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_exporter_recipes_c_custom_ops_fct_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_exporter_recipes_c_custom_ops_fct.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">to_onnx and a custom operator registered with a function</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="export, unflatten and compile =============================">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_torch_export_compile_102_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_torch_export_compile_102.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">102: Tweak onnx export</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Profiles any onnx model on CPU.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_profile_existing_onnx_101_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_profile_existing_onnx_101.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">101: Profile an existing model with onnxruntime</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example shows how to rewrite a graph using a pattern.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_rewrite_101_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_rewrite_101.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">101: Onnx Model Rewriting</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="scikit-learn and torch to train a linear regression.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_torch_linreg_101_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_torch_linreg_101.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">101: Linear Regression and export to ONNX</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example leverages the examples introduced on this page Custom Backends. It uses backend experimental_experiment.torch_dynamo.onnx_custom_backend based on onnxruntime and running on CPU or CUDA. It could easily replaced by experimental_experiment.torch_dynamo.onnx_debug_backend. This one based on the reference implemented from onnx can show the intermediate results if needed. It is very slow.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_torch_custom_backend_101_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_torch_custom_backend_101.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">101: A custom backend for torch</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="torch.export.export behaviour in various situations.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_torch_export_101_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_torch_export_101.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">101: Some dummy examples with torch.export.export</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The convolution is a well known image transformation used to transform an image. It can be used to blur, to compute the gradient in one direction and it is widely used in deep neural networks. Having a fast implementation is important.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_convolutation_matmul_102_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_convolutation_matmul_102.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">102: Convolution and Matrix Multiplication</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example leverages the function torch.compile and the ability to use a custom backend (see Custom Backends) to test the optimization of a model by fusing simple element-wise kernels.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_custom_backend_llama_102_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_custom_backend_llama_102.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">102: Fuse kernels in a small Llama Model</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The script compares the two exporters implemented in pytorch for a part of llama model. The model are compared after all optimizations were made with and onnxruntime.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_llama_diff_export_301_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_llama_diff_export_301.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">301: Compares LLAMA exporters</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The script compares exported models in pytorch using onnxrt backend. It tries to do a side by side of the execution of both models.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_llama_diff_dort_301_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_llama_diff_dort_301.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">301: Compares LLAMA exporters for onnxrt backend</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The script is calling many times the script experimental_experiment.torch_bench.dort_bench.py.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_llama_bench_102_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_llama_bench_102.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">102: Measure LLAMA speed</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="It compares DORT to eager mode and onnxrt backend.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_torch_aot_201_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_torch_aot_201.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">201: Evaluate DORT Training</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="It compares DORT to eager mode and onnxrt backend.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_torch_dort_201_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_torch_dort_201.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">201: Evaluate DORT</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The example evaluates the performance of onnxruntime of a simple torch model after it was converted into ONNX through different processes:">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_torch_export_201_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_torch_export_201.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">201: Evaluate different ways to export a torch model to ONNX</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/plot_onnxscript_102
   /auto_examples/plot_exporter_recipes_oe_custom_ops_inplace
   /auto_examples/plot_exporter_recipes_oe_cond
   /auto_examples/plot_exporter_recipes_c_cond
   /auto_examples/plot_exporter_recipes_oe_custom_ops_fct
   /auto_examples/plot_optimize_101
   /auto_examples/plot_exporter_recipes_c_custom_ops_inplace
   /auto_examples/plot_executorch_102
   /auto_examples/plot_exporter_recipes_c_custom_ops_fct
   /auto_examples/plot_torch_export_compile_102
   /auto_examples/plot_profile_existing_onnx_101
   /auto_examples/plot_rewrite_101
   /auto_examples/plot_torch_linreg_101
   /auto_examples/plot_torch_custom_backend_101
   /auto_examples/plot_torch_export_101
   /auto_examples/plot_convolutation_matmul_102
   /auto_examples/plot_custom_backend_llama_102
   /auto_examples/plot_llama_diff_export_301
   /auto_examples/plot_llama_diff_dort_301
   /auto_examples/plot_llama_bench_102
   /auto_examples/plot_torch_aot_201
   /auto_examples/plot_torch_dort_201
   /auto_examples/plot_torch_export_201


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: auto_examples_python.zip </auto_examples/auto_examples_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: auto_examples_jupyter.zip </auto_examples/auto_examples_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
