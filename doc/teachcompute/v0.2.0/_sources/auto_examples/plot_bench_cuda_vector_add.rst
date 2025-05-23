
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_examples/plot_bench_cuda_vector_add.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_examples_plot_bench_cuda_vector_add.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_plot_bench_cuda_vector_add.py:


.. _l-example-cuda-vector-addition:

Measuring CUDA performance with a vector addition
=================================================

Measure the time between two additions, one with CUDA, one with
:epkg:`numpy`. The script can be profiled with
:epkg:`Nsight`.

::

    nsys profile python _doc/examples/plot_bench_cuda_vector_add.py

Vector Add
++++++++++

.. GENERATED FROM PYTHON SOURCE LINES 18-83

.. code-block:: Python


    from tqdm import tqdm
    import numpy
    import matplotlib.pyplot as plt
    from pandas import DataFrame
    from teachcompute.ext_test_case import measure_time, unit_test_going
    import torch

    has_cuda = torch.cuda.is_available()

    try:
        from teachcompute.validation.cuda.cuda_example_py import vector_add
    except ImportError:
        has_cuda = False


    def cuda_vector_add(values):
        torch.cuda.nvtx.range_push(f"CUDA dim={values.size}")
        res = vector_add(values, values, 0)
        torch.cuda.nvtx.range_pop()
        return res


    obs = []
    dims = [2**10, 2**15, 2**20, 2**25]
    if unit_test_going():
        dims = [10, 20, 30]
    for dim in tqdm(dims):
        values = numpy.ones((dim,), dtype=numpy.float32).ravel()

        if has_cuda:
            diff = numpy.abs(vector_add(values, values, 0) - (values + values)).max()
            res = measure_time(lambda values=values: cuda_vector_add(values), max_time=0.5)

            obs.append(
                dict(
                    dim=dim,
                    size=values.size,
                    time=res["average"],
                    fct="CUDA",
                    time_per_element=res["average"] / dim,
                    diff=diff,
                )
            )

        diff = 0
        res = measure_time(lambda values=values: values + values, max_time=0.5)

        obs.append(
            dict(
                dim=dim,
                size=values.size,
                time=res["average"],
                fct="numpy",
                time_per_element=res["average"] / dim,
                diff=0,
            )
        )


    df = DataFrame(obs)
    piv = df.pivot(index="dim", columns="fct", values="time_per_element")
    print(piv)






.. rst-class:: sphx-glr-script-out

 .. code-block:: none

      0%|          | 0/4 [00:00<?, ?it/s]     25%|██▌       | 1/4 [00:03<00:10,  3.50s/it]     50%|█████     | 2/4 [00:04<00:04,  2.14s/it]     75%|███████▌  | 3/4 [00:05<00:01,  1.70s/it]    100%|██████████| 4/4 [00:08<00:00,  2.15s/it]    100%|██████████| 4/4 [00:08<00:00,  2.17s/it]
    fct               CUDA         numpy
    dim                                 
    1024      1.200411e-06  6.429443e-10
    32768     4.023615e-08  1.519021e-10
    1048576   1.445003e-08  4.195622e-10
    33554432  1.171827e-08  1.075832e-09




.. GENERATED FROM PYTHON SOURCE LINES 84-86

Plots
+++++

.. GENERATED FROM PYTHON SOURCE LINES 86-97

.. code-block:: Python


    piv_diff = df.pivot(index="dim", columns="fct", values="diff")
    piv_time = df.pivot(index="dim", columns="fct", values="time")

    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    piv.plot(ax=ax[0], logx=True, title="Comparison between two summation")
    piv_diff.plot(ax=ax[1], logx=True, logy=True, title="Summation errors")
    piv_time.plot(ax=ax[2], logx=True, logy=True, title="Total time")
    fig.tight_layout()
    fig.savefig("plot_bench_cuda_vector_add.png")




.. image-sg:: /auto_examples/images/sphx_glr_plot_bench_cuda_vector_add_001.png
   :alt: Comparison between two summation, Summation errors, Total time
   :srcset: /auto_examples/images/sphx_glr_plot_bench_cuda_vector_add_001.png
   :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    /home/xadupre/vv/this312/lib/python3.12/site-packages/pandas/plotting/_matplotlib/core.py:822: UserWarning: Data has no positive values, and therefore cannot be log-scaled.
      labels = axis.get_majorticklabels() + axis.get_minorticklabels()




.. GENERATED FROM PYTHON SOURCE LINES 98-103

CUDA seems very slow but in fact, all the time is spent
in moving the data from the CPU memory (Host) to the GPU memory (device).

.. image:: ../images/nsight_vector_add.png



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 13.921 seconds)


.. _sphx_glr_download_auto_examples_plot_bench_cuda_vector_add.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_bench_cuda_vector_add.ipynb <plot_bench_cuda_vector_add.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_bench_cuda_vector_add.py <plot_bench_cuda_vector_add.py>`

    .. container:: sphx-glr-download sphx-glr-download-zip

      :download:`Download zipped: plot_bench_cuda_vector_add.zip <plot_bench_cuda_vector_add.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
