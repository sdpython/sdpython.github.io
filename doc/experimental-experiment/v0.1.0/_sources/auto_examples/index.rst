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

    git clone https://github.com/xadupre/experimental-experiment.git
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


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example shows how to optimize a graph using pattern optimization. The graph was obtained b...">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_optimize_101_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_optimize_101.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">101: Graph Optimization</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="scikit-learn and torch to train a linear regression.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_torch_linreg_101_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_torch_linreg_101.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">101: Linear Regression and export to ONNX</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example leverages the examples introduced on this page Custom Backends. It uses backend ex...">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_torch_custom_backend_101_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_torch_custom_backend_101.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">101: A custom backend for torch</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The convolution is a well known image transformation used to transform an image. It can be used...">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_convolutation_matmul_102_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_convolutation_matmul_102.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">102: Convolution and Matrix Multiplication</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The script compares the two exporters implemented in pytorch for a part of llama model. The mod...">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_llama_diff_export_301_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_llama_diff_export_301.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">301: Compares LLAMA exporters</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="The script compares exported models in pytorch using onnxrt backend. It tries to do a side by s...">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_llama_diff_dort_301_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_llama_diff_dort_301.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">301: Compares LLAMA exporters for onnxrt backend</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="The example evaluates the performance of onnxruntime of a simple torch model after it was conve...">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_torch_export_201_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_torch_export_201.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">201: Evaluate different ways to export a torch model to ONNX</div>
    </div>


.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/plot_optimize_101
   /auto_examples/plot_profile_existing_onnx_101
   /auto_examples/plot_torch_linreg_101
   /auto_examples/plot_torch_custom_backend_101
   /auto_examples/plot_convolutation_matmul_102
   /auto_examples/plot_llama_diff_export_301
   /auto_examples/plot_llama_bench_102
   /auto_examples/plot_llama_diff_dort_301
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
