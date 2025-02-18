
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_examples/plot_benchmark_parallel_process_concurrent.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_examples_plot_benchmark_parallel_process_concurrent.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_plot_benchmark_parallel_process_concurrent.py:


Parallelization of a dot product with processes (concurrent.futures)
====================================================================

Uses processes to parallelize a dot product is not
a very solution because processes do not share memory,
they need to exchange data. This parallelisation
is efficient if the ratio *exchanged data / computation time*
is low. This example uses :epkg:`concurrent.futures`.
The cost of creating new processes is also significant.

.. GENERATED FROM PYTHON SOURCE LINES 12-40

.. code-block:: Python


    import numpy
    from tqdm import tqdm
    from pandas import DataFrame
    import matplotlib.pyplot as plt
    import concurrent.futures as cf
    from teachcompute.ext_test_case import measure_time


    def parallel_numpy_dot(va, vb, max_workers=2):
        if max_workers == 2:
            with cf.ThreadPoolExecutor(max_workers=max_workers) as e:
                m = va.shape[0] // 2
                f1 = e.submit(numpy.dot, va[:m], vb[:m])
                f2 = e.submit(numpy.dot, va[m:], vb[m:])
                return f1.result() + f2.result()
        elif max_workers == 3:
            with cf.ThreadPoolExecutor(max_workers=max_workers) as e:
                m = va.shape[0] // 3
                m2 = va.shape[0] * 2 // 3
                f1 = e.submit(numpy.dot, va[:m], vb[:m])
                f2 = e.submit(numpy.dot, va[m:m2], vb[m:m2])
                f3 = e.submit(numpy.dot, va[m2:], vb[m2:])
                return f1.result() + f2.result() + f3.result()
        else:
            raise NotImplementedError()









.. GENERATED FROM PYTHON SOURCE LINES 41-42

We check that it returns the same values.

.. GENERATED FROM PYTHON SOURCE LINES 42-49

.. code-block:: Python



    va = numpy.random.randn(100).astype(numpy.float64)
    vb = numpy.random.randn(100).astype(numpy.float64)
    print(parallel_numpy_dot(va, vb), numpy.dot(va, vb))






.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    -11.963328675229187 -11.963328675229185




.. GENERATED FROM PYTHON SOURCE LINES 50-51

Let's benchmark.

.. GENERATED FROM PYTHON SOURCE LINES 51-65

.. code-block:: Python

    res = []
    for n in tqdm([100000, 1000000, 10000000, 100000000]):
        va = numpy.random.randn(n).astype(numpy.float64)
        vb = numpy.random.randn(n).astype(numpy.float64)

        m1 = measure_time("dot(va, vb, 2)", dict(va=va, vb=vb, dot=parallel_numpy_dot))
        m2 = measure_time("dot(va, vb)", dict(va=va, vb=vb, dot=numpy.dot))
        res.append({"N": n, "numpy.dot": m2["average"], "futures": m1["average"]})

    df = DataFrame(res).set_index("N")
    print(df)
    df.plot(logy=True, logx=True)
    plt.title("Parallel / numpy dot")




.. image-sg:: /auto_examples/images/sphx_glr_plot_benchmark_parallel_process_concurrent_001.png
   :alt: Parallel / numpy dot
   :srcset: /auto_examples/images/sphx_glr_plot_benchmark_parallel_process_concurrent_001.png
   :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 .. code-block:: none

      0%|          | 0/4 [00:00<?, ?it/s]     25%|██▌       | 1/4 [00:16<00:48, 16.19s/it]     50%|█████     | 2/4 [00:23<00:21, 10.87s/it]     75%|███████▌  | 3/4 [00:33<00:10, 10.69s/it]    100%|██████████| 4/4 [01:16<00:00, 23.16s/it]    100%|██████████| 4/4 [01:16<00:00, 19.02s/it]
               numpy.dot   futures
    N                             
    100000      0.002819  0.029472
    1000000     0.002338  0.011743
    10000000    0.007796  0.012140
    100000000   0.035683  0.038637

    Text(0.5, 1.0, 'Parallel / numpy dot')



.. GENERATED FROM PYTHON SOURCE LINES 66-68

The parallelisation is inefficient
unless the vectors are big.


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (1 minutes 16.329 seconds)


.. _sphx_glr_download_auto_examples_plot_benchmark_parallel_process_concurrent.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_benchmark_parallel_process_concurrent.ipynb <plot_benchmark_parallel_process_concurrent.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_benchmark_parallel_process_concurrent.py <plot_benchmark_parallel_process_concurrent.py>`

    .. container:: sphx-glr-download sphx-glr-download-zip

      :download:`Download zipped: plot_benchmark_parallel_process_concurrent.zip <plot_benchmark_parallel_process_concurrent.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
