:orphan:


Gallerie d'exemples
===================



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Uses processes to parallelize a dot product is not a very solution because processes do not share memory, they need to exchange data. This parallelisation is efficient if the ratio exchanged data / computation time is low. joblib is used by scikit-learn. The cost of creating new processes is also significant.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_benchmark_long_parallel_process_joblib_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_benchmark_long_parallel_process_joblib.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Parallelization of a dot product with processes (joblib)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="numpy has a very fast implementation of matrix multiplication. There are many ways to be slower. The following uses timeit to compare implementations.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_benchmark_dot_mul_timeit_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_benchmark_dot_mul_timeit.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Compares matrix multiplication implementations with timeit</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The matrix multiplication m1 @ m2 @ m3 can be done in two different ways: (m1 @ m2) @ m3 or m1 @ (m2 @ m3). Are these two orders equivalent or is there a better order?">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_benchmark_associative_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_benchmark_associative.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Associativity and matrix multiplication</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Uses processes to parallelize a dot product is not a very solution because processes do not share memory, they need to exchange data. This parallelisation is efficient if the ratio exchanged data / computation time is low. This example uses concurrent.futures. The cost of creating new processes is also significant.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_benchmark_parallel_process_concurrent_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_benchmark_parallel_process_concurrent.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Parallelization of a dot product with processes (concurrent.futures)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="numpy has a very fast implementation of the dot product. It is difficult to be better and very easy to be slower. This example looks into a couple of slower implementations.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_benchmark_dot_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_benchmark_dot.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Compares dot implementations (numpy, python, blas)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The example compares the time spend in computing the sum of all coefficients of a matrix when the function walks through the coefficients by rows or by columns.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_bench_cpu_vector_sum_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_bench_cpu_vector_sum.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Measuring CPU performance with a vector sum</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The benchmark looks into different ways to implement thresholding: every value of a vector superior to mx is replaced by mx (numpy.clip). It compares several implementation to numpy.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_benchmark_filter_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_benchmark_filter.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Compares filtering implementations (numpy, cython)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Measure the time between two additions, one with CUDA, one with numpy. The script can be profiled with Nsight.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_bench_cuda_vector_add_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_bench_cuda_vector_add.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Measuring CUDA performance with a vector addition</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Parallelization usually means a summation is done with a random order. That may lead to different values if the computation is made many times even though the result should be the same. This example compares summation of random permutation of the same array of values.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_check_random_order_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_check_random_order.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Random order for a sum</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="numpy has a very fast implementation of the dot product. It is difficult to be better and very easy to be slower. This example looks into a couple of slower implementations with cython. The tested functions are the following:">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_benchmark_dot_cython_omp_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_benchmark_dot_cython_omp.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Compares dot implementations (numpy, c++, sse, openmp)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The objective is to measure the summation of all elements from a tensor.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_bench_cuda_vector_sum_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_bench_cuda_vector_sum.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Measuring CUDA performance with a vector sum</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The example compares the time spend in computing the sum of all coefficients of a matrix when the function walks through the coefficients by rows or by columns when the computation is parallelized.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_bench_cpu_vector_sum_parallel_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_bench_cpu_vector_sum_parallel.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Measuring CPU performance with a parallelized vector sum</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A piecewise linear function is implemented and trained following the tutorial Custom C++ and CUDA Extensions.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_piecewise_linear_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_piecewise_linear.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Compares implementations for a Piecewise Linear</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Measure the time between two additions, with or without streams. The script can be profiled with Nsight.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_bench_cuda_vector_add_stream_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_bench_cuda_vector_add_stream.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Measuring CUDA performance with a vector addition with streams</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="numpy has a very fast implementation of the dot product. It is difficult to be better and very easy to be slower. This example looks into a couple of slower implementations with cython. The tested functions are the following:">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_benchmark_dot_cython_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_benchmark_dot_cython.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Compares dot implementations (numpy, cython, c++, sse)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The example compares the time spend in computing the sum of all coefficients of a matrix when the function walks through the coefficients by rows or by columns when the computation is parallelized or uses AVX instructions.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_bench_cpu_vector_sum_avx_parallel_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_bench_cpu_vector_sum_avx_parallel.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Measuring CPU performance with a parallelized vector sum and AVX</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script does not export a full llama model but a shorter one to be able to fast iterate on improvments. See LlamaConfig. The model is then converted into ONNX. It can be seen with Netron which can be also used through a VS Code Extension.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_export_model_onnx_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_export_model_onnx.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Export a LLAMA model into ONNX</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="numpy has a very fast implementation of matrix multiplication. There are many ways to be slower.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_benchmark_dot_mul_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_benchmark_dot_mul.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Compares matrix multiplication implementations</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Le notebook explore différentes façons de sérialiser des données et leurs limites.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_serialisation_examples_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_serialisation_examples.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Sérialisation</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/plot_benchmark_long_parallel_process_joblib
   /auto_examples/plot_benchmark_dot_mul_timeit
   /auto_examples/plot_benchmark_associative
   /auto_examples/plot_benchmark_parallel_process_concurrent
   /auto_examples/plot_benchmark_dot
   /auto_examples/plot_bench_cpu_vector_sum
   /auto_examples/plot_benchmark_filter
   /auto_examples/plot_bench_cuda_vector_add
   /auto_examples/plot_check_random_order
   /auto_examples/plot_benchmark_dot_cython_omp
   /auto_examples/plot_bench_cuda_vector_sum
   /auto_examples/plot_bench_cpu_vector_sum_parallel
   /auto_examples/plot_piecewise_linear
   /auto_examples/plot_bench_cuda_vector_add_stream
   /auto_examples/plot_benchmark_dot_cython
   /auto_examples/plot_bench_cpu_vector_sum_avx_parallel
   /auto_examples/plot_export_model_onnx
   /auto_examples/plot_benchmark_dot_mul
   /auto_examples/plot_serialisation_examples


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: auto_examples_python.zip </auto_examples/auto_examples_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: auto_examples_jupyter.zip </auto_examples/auto_examples_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
