:orphan:

.. _l-example-gallery:

Examples Gallery
================







.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Processor caches must be taken into account when writing an algorithm, see Memory part 2: CPU caches from Ulrich Drepper.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_bench_cpu_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_bench_cpu.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Measuring CPU performance</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="onnx-extended includes an implementation of operator Conv in language C++ must faster than the python implementation available in package onnx. These implementations are automatically available through class onnx_extended.reference.CReferenceEvaluator. The following example compares the processing time for three runtimes.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_op_conv_py_vs_c_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_op_conv_py_vs_c.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Using C implementation of operator Conv</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The following code measures the performance of the python bindings against a cython binding. The time spent in it is not significant when the computation is huge but it may be for small matrices.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_bench_cypy_ort_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_bench_cypy_ort.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Measuring onnxruntime performance against a cython binding</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Whenever computing the prediction of a tree with a sparse tensor, is it faster to density first and then to compute the prediction or to keep the tensor in its sparse representation and do look up? The parameter nrnd can be seen as the depth of a tree.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_bench_sparse_access_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_bench_sparse_access.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Evaluating random access for sparse</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The banchmark measures the performance of a TfIdfVectizer along two parameters, the vocabulary size, the batch size whether. It measures the benefit of using sparse implementation through the computation time and the memory peak.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_op_tfidfvectorizer_sparse_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_op_tfidfvectorizer_sparse.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Measuring performance of TfIdfVectorizer</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This benchmark looks into various combinations allowed by functions cublasLtMatmul. The tested configurations are available at cuda_gemm.cu.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_bench_gemm_f8_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_bench_gemm_f8.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Measuring Gemm performance with different input and output tests</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="One big Gemm or two smaller gemm?">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_op_gemm2_cuda_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_op_gemm2_cuda.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gemm Exploration with CUDA</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This configuration happens in a Llama model.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_op_transpose_2d_cast_cuda_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_op_transpose_2d_cast_cuda.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Fuse Tranpose and Cast on CUDA</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example compares different equations for function numpy.einsum. It compares numpy implementation to a custom implementation, onnxruntime implementation and opt-einsum optimisation. If available, tensorflow and pytorch are included as well. The custom implementation does not do any transpose. It uses parallelisation and SIMD optimization when the summation happens on the last axis of both matrices. It only implements matrix multiplication. We also measure the improvment made with function onnx_extended.tools.einsum.einsum_fct.einsum.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_op_einsum_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_op_einsum.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Compares implementations of Einsum</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The examples compare the performaance of two fused operators Mul with the unfused sequence.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_op_mul_cuda_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_op_mul_cuda.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Fusing multiplication operators on CUDA</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An example with Conv. The floats followed the IEEE standard Single-precision floating-point format. The number is interprated in a different whether the exponent is null or not. When it is null, it is called a denormalized number or subnormal number. Let&#x27;s see their impact on the computation time through the operator Conv.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_op_conv_denorm_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_op_conv_denorm.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">How float format has an impact on speed computation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The execution of a TreeEnsembleRegressor can lead to very different results depending on how the computation is parallelized. By trees, by rows, by both, for only one row, for a short batch of rows, a longer one. The implementation in onnxruntime does not let the user changed the predetermined settings but a custom kernel might. That&#x27;s what this example is measuring.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_op_tree_ensemble_optim_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_op_tree_ensemble_optim.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">TreeEnsemble optimization</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="How to parallelize something like the following?">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_op_scatternd_cuda_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_op_scatternd_cuda.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Optimizing ScatterND operator on CUDA</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="How to parallelize something like the following?">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_op_scatternd_mask_cuda_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_op_scatternd_mask_cuda.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Optimizing Masked ScatterND operator on CUDA</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The example benchmarks the sparse implementation for TreeEnsemble. The default set of optimized parameters is very short and is meant to be executed fast. Many more parameters can be tried.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_op_tree_ensemble_sparse_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_op_tree_ensemble_sparse.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">TreeEnsemble, dense, and sparse</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The benchmark profiles the execution of Gemm for different types and configuration. That includes a custom operator only available on CUDA calling function cublasLtMatmul.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_profile_gemm_ort_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_profile_gemm_ort.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Profiles a simple onnx graph including a singleGemm</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The benchmark measures the performance of Gemm for different types and configuration. That includes a custom operator only available on CUDA calling function cublasLtMatmul. This function offers many options.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_bench_gemm_ort_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_bench_gemm_ort.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Measuring performance about Gemm with onnxruntime</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This is a simplified benchmark to compare TreeEnsemble implementations (see below) Run python plot_op_tree_ensemble_implementations.py --help to change the tree dimension. Here are the following implementation:">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_op_tree_ensemble_implementations_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_op_tree_ensemble_implementations.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Evaluate different implementation of TreeEnsemble</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/plot_bench_cpu
   /auto_examples/plot_op_conv_py_vs_c
   /auto_examples/plot_bench_cypy_ort
   /auto_examples/plot_bench_sparse_access
   /auto_examples/plot_op_tfidfvectorizer_sparse
   /auto_examples/plot_bench_gemm_f8
   /auto_examples/plot_op_gemm2_cuda
   /auto_examples/plot_op_transpose_2d_cast_cuda
   /auto_examples/plot_op_einsum
   /auto_examples/plot_op_mul_cuda
   /auto_examples/plot_op_conv_denorm
   /auto_examples/plot_op_tree_ensemble_optim
   /auto_examples/plot_op_scatternd_cuda
   /auto_examples/plot_op_scatternd_mask_cuda
   /auto_examples/plot_op_tree_ensemble_sparse
   /auto_examples/plot_profile_gemm_ort
   /auto_examples/plot_bench_gemm_ort
   /auto_examples/plot_op_tree_ensemble_implementations


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: auto_examples_python.zip </auto_examples/auto_examples_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: auto_examples_jupyter.zip </auto_examples/auto_examples_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
