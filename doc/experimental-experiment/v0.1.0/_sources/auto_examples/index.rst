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

Then install *onnx-rewriter*.

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

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_optimize_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_optimize.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Graph Optimization</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Profiles any onnx model on CPU.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_profile_existing_onnx_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_profile_existing_onnx.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Profile an existing model</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="scikit-learn and torch to train a linear regression.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_torch_linreg_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_torch_linreg.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Linear Regression</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example leverages the examples introduced on this page Custom Backends. It uses backend ex...">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_torch_custom_backend_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_torch_custom_backend.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A custom backend for torch</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The convolution is a well known image transformation used to transform an image. It can be used...">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_convolutation_matmul_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_convolutation_matmul.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Convolution and Matrix Multiplication</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The script compares the two exporters implemented in pytorch for a part of llama model. The mod...">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_llama_diff_export_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_llama_diff_export.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Compares LLAMA exporters</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The script is calling many times the script experimental_experiment.torch_bench.dort_bench.py.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_llama_bench_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_llama_bench.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Measure LLAMA speed</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The script compares exported models in pytorch using onnxrt backend. It tries to do a side by s...">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_llama_diff_dort_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_llama_diff_dort.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Compares LLAMA exporters for onnxrt backend</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="It compares DORT to eager mode and onnxrt backend.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_torch_dort_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_torch_dort.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Evaluate DORT</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="It compares DORT to eager mode and onnxrt backend.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_torch_aot_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_torch_aot.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Evaluate DORT Training</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The example evaluates the performance of onnxruntime of a simple torch model after it was conve...">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_torch_export_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_torch_export.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Evaluate different ways to export a torch model to ONNX</div>
    </div>


.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/plot_optimize
   /auto_examples/plot_profile_existing_onnx
   /auto_examples/plot_torch_linreg
   /auto_examples/plot_torch_custom_backend
   /auto_examples/plot_convolutation_matmul
   /auto_examples/plot_llama_diff_export
   /auto_examples/plot_llama_bench
   /auto_examples/plot_llama_diff_dort
   /auto_examples/plot_torch_dort
   /auto_examples/plot_torch_aot
   /auto_examples/plot_torch_export


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: auto_examples_python.zip </auto_examples/auto_examples_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: auto_examples_jupyter.zip </auto_examples/auto_examples_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
