
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_examples/plot_bench_cpu_vector_sum_avx_parallel.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_examples_plot_bench_cpu_vector_sum_avx_parallel.py>`
        to download the full example code

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_plot_bench_cpu_vector_sum_avx_parallel.py:


Measuring CPU performance with a parallelized vector sum and AVX
================================================================

The example compares the time spend in computing the sum of all
coefficients of a matrix when the function walks through the coefficients
by rows or by columns when the computation is parallelized or uses
AVX instructions.

Vector Sum
++++++++++

.. GENERATED FROM PYTHON SOURCE LINES 13-93

.. code-block:: Python

    from tqdm import tqdm
    import numpy
    import matplotlib.pyplot as plt
    from pandas import DataFrame
    from onnx_extended.ext_test_case import measure_time, unit_test_going
    from onnx_extended.validation.cpu._validation import (
        vector_sum_array as vector_sum,
        vector_sum_array_parallel as vector_sum_parallel,
        vector_sum_array_avx as vector_sum_avx,
        vector_sum_array_avx_parallel as vector_sum_avx_parallel,
    )

    obs = []
    dims = [500, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 2000]
    if unit_test_going():
        dims = dims[:2]
    for dim in tqdm(dims):
        values = numpy.ones((dim, dim), dtype=numpy.float32).ravel()
        diff = abs(vector_sum(dim, values, True) - dim**2)

        res = measure_time(lambda: vector_sum(dim, values, True), max_time=0.5)

        obs.append(
            dict(
                dim=dim,
                size=values.size,
                time=res["average"],
                direction="rows",
                time_per_element=res["average"] / dim**2,
                diff=diff,
            )
        )

        res = measure_time(lambda: vector_sum_parallel(dim, values, True), max_time=0.5)

        obs.append(
            dict(
                dim=dim,
                size=values.size,
                time=res["average"],
                direction="rows//",
                time_per_element=res["average"] / dim**2,
                diff=diff,
            )
        )

        diff = abs(vector_sum_avx(dim, values) - dim**2)
        res = measure_time(lambda: vector_sum_avx(dim, values), max_time=0.5)

        obs.append(
            dict(
                dim=dim,
                size=values.size,
                time=res["average"],
                direction="avx",
                time_per_element=res["average"] / dim**2,
                diff=diff,
            )
        )

        diff = abs(vector_sum_avx_parallel(dim, values) - dim**2)
        res = measure_time(lambda: vector_sum_avx_parallel(dim, values), max_time=0.5)

        obs.append(
            dict(
                dim=dim,
                size=values.size,
                time=res["average"],
                direction="avx//",
                time_per_element=res["average"] / dim**2,
                diff=diff,
            )
        )


    df = DataFrame(obs)
    piv = df.pivot(index="dim", columns="direction", values="time_per_element")
    print(piv)






.. rst-class:: sphx-glr-script-out

 .. code-block:: none

      0%|          | 0/14 [00:00<?, ?it/s]      7%|▋         | 1/14 [00:06<01:18,  6.00s/it]     14%|█▍        | 2/14 [00:09<00:56,  4.69s/it]     21%|██▏       | 3/14 [00:12<00:43,  3.93s/it]     29%|██▊       | 4/14 [00:18<00:46,  4.64s/it]     36%|███▌      | 5/14 [00:22<00:38,  4.24s/it]     43%|████▎     | 6/14 [00:27<00:37,  4.73s/it]     50%|█████     | 7/14 [00:34<00:37,  5.39s/it]     57%|█████▋    | 8/14 [00:39<00:31,  5.17s/it]     64%|██████▍   | 9/14 [00:42<00:22,  4.44s/it]     71%|███████▏  | 10/14 [00:46<00:18,  4.55s/it]     79%|███████▊  | 11/14 [00:50<00:13,  4.40s/it]     86%|████████▌ | 12/14 [00:53<00:07,  3.83s/it]     93%|█████████▎| 13/14 [00:55<00:03,  3.35s/it]    100%|██████████| 14/14 [01:00<00:00,  3.77s/it]    100%|██████████| 14/14 [01:00<00:00,  4.31s/it]
    direction           avx         avx//          rows        rows//
    dim                                                              
    500        2.834010e-10  4.159500e-10  6.483628e-09  1.673606e-08
    700        3.440244e-10  6.292865e-10  2.236295e-09  1.643943e-09
    800        3.890547e-10  2.895110e-10  2.746110e-09  9.074247e-10
    900        1.088019e-09  1.811715e-09  2.095698e-09  8.613172e-09
    1000       7.157063e-10  3.627104e-10  3.239336e-09  5.666816e-09
    1100       6.915556e-10  4.884803e-10  3.203873e-09  2.006794e-09
    1200       9.628557e-10  2.438816e-09  2.613826e-09  3.812641e-09
    1300       6.958355e-10  1.350762e-09  2.769874e-09  3.778286e-09
    1400       5.839910e-10  1.572937e-09  4.716569e-09  4.709725e-09
    1500       1.131952e-09  1.962012e-09  2.766701e-09  4.432219e-09
    1600       1.292369e-09  2.508789e-09  2.244841e-09  2.418712e-09
    1700       1.606061e-09  3.678123e-09  3.437443e-09  3.933195e-09
    1800       1.186381e-09  3.321332e-09  3.571899e-09  4.165062e-09
    2000       8.506587e-10  2.360019e-09  4.186000e-09  2.929390e-09




.. GENERATED FROM PYTHON SOURCE LINES 94-96

Plots
+++++

.. GENERATED FROM PYTHON SOURCE LINES 96-107

.. code-block:: Python


    piv_diff = df.pivot(index="dim", columns="direction", values="diff")
    piv_time = df.pivot(index="dim", columns="direction", values="time")

    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    piv.plot(ax=ax[0], logx=True, title="Comparison between two summation")
    piv_diff.plot(ax=ax[1], logx=True, logy=True, title="Summation errors")
    piv_time.plot(ax=ax[2], logx=True, logy=True, title="Total time")
    fig.tight_layout()
    fig.savefig("plot_bench_cpu_vector_sum_avx_parallel.png")




.. image-sg:: /auto_examples/images/sphx_glr_plot_bench_cpu_vector_sum_avx_parallel_001.png
   :alt: Comparison between two summation, Summation errors, Total time
   :srcset: /auto_examples/images/sphx_glr_plot_bench_cpu_vector_sum_avx_parallel_001.png
   :class: sphx-glr-single-img





.. GENERATED FROM PYTHON SOURCE LINES 108-109

AVX is faster.


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (1 minutes 4.464 seconds)


.. _sphx_glr_download_auto_examples_plot_bench_cpu_vector_sum_avx_parallel.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_bench_cpu_vector_sum_avx_parallel.ipynb <plot_bench_cpu_vector_sum_avx_parallel.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_bench_cpu_vector_sum_avx_parallel.py <plot_bench_cpu_vector_sum_avx_parallel.py>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
