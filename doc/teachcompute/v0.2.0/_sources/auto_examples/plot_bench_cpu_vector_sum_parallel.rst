
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_examples/plot_bench_cpu_vector_sum_parallel.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_examples_plot_bench_cpu_vector_sum_parallel.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_plot_bench_cpu_vector_sum_parallel.py:


Measuring CPU performance with a parallelized vector sum
========================================================

The example compares the time spend in computing the sum of all
coefficients of a matrix when the function walks through the coefficients
by rows or by columns when the computation is parallelized.

Vector Sum
++++++++++

.. GENERATED FROM PYTHON SOURCE LINES 12-85

.. code-block:: Python


    from tqdm import tqdm
    import numpy
    import matplotlib.pyplot as plt
    from pandas import DataFrame
    from teachcompute.ext_test_case import measure_time, unit_test_going
    from teachcompute.validation.cpu._validation import (
        vector_sum_array as vector_sum,
        vector_sum_array_parallel as vector_sum_parallel,
    )

    obs = []
    dims = [500, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 2000]
    if unit_test_going():
        dims = [10, 20]
    for dim in tqdm(dims):
        values = numpy.ones((dim, dim), dtype=numpy.float32).ravel()
        diff = abs(vector_sum(dim, values, True) - dim**2)

        res = measure_time(
            lambda dim=dim, values=values: vector_sum(dim, values, True), max_time=0.5
        )

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

        res = measure_time(
            lambda dim=dim, values=values: vector_sum_parallel(dim, values, True),
            max_time=0.5,
        )

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

        diff = abs(vector_sum(dim, values, False) - dim**2)
        res = measure_time(
            lambda dim=dim, values=values: vector_sum_parallel(dim, values, False),
            max_time=0.5,
        )

        obs.append(
            dict(
                dim=dim,
                size=values.size,
                time=res["average"],
                direction="cols//",
                time_per_element=res["average"] / dim**2,
                diff=diff,
            )
        )


    df = DataFrame(obs)
    piv = df.pivot(index="dim", columns="direction", values="time_per_element")
    print(piv)






.. rst-class:: sphx-glr-script-out

 .. code-block:: none

      0%|          | 0/14 [00:00<?, ?it/s]      7%|▋         | 1/14 [00:01<00:22,  1.70s/it]     14%|█▍        | 2/14 [00:03<00:20,  1.72s/it]     21%|██▏       | 3/14 [00:05<00:18,  1.72s/it]     29%|██▊       | 4/14 [00:07<00:17,  1.79s/it]     36%|███▌      | 5/14 [00:08<00:15,  1.76s/it]     43%|████▎     | 6/14 [00:11<00:15,  1.96s/it]     50%|█████     | 7/14 [00:12<00:13,  1.94s/it]     57%|█████▋    | 8/14 [00:14<00:11,  1.90s/it]     64%|██████▍   | 9/14 [00:16<00:09,  1.88s/it]     71%|███████▏  | 10/14 [00:18<00:07,  1.84s/it]     79%|███████▊  | 11/14 [00:20<00:05,  1.82s/it]     86%|████████▌ | 12/14 [00:21<00:03,  1.78s/it]     93%|█████████▎| 13/14 [00:23<00:01,  1.74s/it]    100%|██████████| 14/14 [00:25<00:00,  1.75s/it]    100%|██████████| 14/14 [00:25<00:00,  1.80s/it]
    direction        cols//          rows        rows//
    dim                                                
    500        1.294664e-08  9.209370e-10  4.666173e-08
    700        1.518723e-08  1.149378e-09  2.897091e-08
    800        7.104689e-09  9.410522e-10  1.562958e-08
    900        4.572897e-09  9.831829e-10  1.326127e-08
    1000       2.417810e-09  8.962873e-10  1.603054e-08
    1100       9.263096e-10  9.648644e-10  1.887751e-09
    1200       1.968152e-09  9.654317e-10  4.819424e-09
    1300       1.028106e-09  1.037502e-09  1.357359e-09
    1400       1.164685e-09  1.039371e-09  1.737098e-09
    1500       1.234987e-09  9.843251e-10  1.272064e-09
    1600       1.100550e-09  1.064941e-09  9.647264e-10
    1700       1.264651e-09  9.763337e-10  1.862488e-09
    1800       1.071664e-09  9.563353e-10  1.512979e-09
    2000       1.257330e-09  1.164962e-09  9.101863e-10




.. GENERATED FROM PYTHON SOURCE LINES 86-88

Plots
+++++

.. GENERATED FROM PYTHON SOURCE LINES 88-99

.. code-block:: Python


    piv_diff = df.pivot(index="dim", columns="direction", values="diff")
    piv_time = df.pivot(index="dim", columns="direction", values="time")

    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    piv.plot(ax=ax[0], logx=True, title="Comparison between two summation")
    piv_diff.plot(ax=ax[1], logx=True, logy=True, title="Summation errors")
    piv_time.plot(ax=ax[2], logx=True, logy=True, title="Total time")
    fig.tight_layout()
    fig.savefig("plot_bench_cpu_vector_sum_parallel.png")




.. image-sg:: /auto_examples/images/sphx_glr_plot_bench_cpu_vector_sum_parallel_001.png
   :alt: Comparison between two summation, Summation errors, Total time
   :srcset: /auto_examples/images/sphx_glr_plot_bench_cpu_vector_sum_parallel_001.png
   :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    /home/xadupre/vv/this/lib/python3.10/site-packages/pandas/plotting/_matplotlib/core.py:822: UserWarning: Data has no positive values, and therefore cannot be log-scaled.
      labels = axis.get_majorticklabels() + axis.get_minorticklabels()




.. GENERATED FROM PYTHON SOURCE LINES 100-104

The summation by rows is much faster as expected.
That explains why it is usually more efficient to
transpose the first matrix before a matrix multiplication.
Parallelization is faster.


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 26.553 seconds)


.. _sphx_glr_download_auto_examples_plot_bench_cpu_vector_sum_parallel.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_bench_cpu_vector_sum_parallel.ipynb <plot_bench_cpu_vector_sum_parallel.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_bench_cpu_vector_sum_parallel.py <plot_bench_cpu_vector_sum_parallel.py>`

    .. container:: sphx-glr-download sphx-glr-download-zip

      :download:`Download zipped: plot_bench_cpu_vector_sum_parallel.zip <plot_bench_cpu_vector_sum_parallel.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
