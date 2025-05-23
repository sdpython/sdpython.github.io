
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_examples/plot_benchmark_filter.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_examples_plot_benchmark_filter.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_plot_benchmark_filter.py:


.. _l-compare-filtering-implementation:

Compares filtering implementations (numpy, cython)
==================================================

The benchmark looks into different ways to implement
thresholding: every value of a vector superior to *mx*
is replaced by *mx* (:func:`numpy.clip`).
It compares several implementation to :epkg:`numpy`.

* :func:`cfilter_dmax <teachcompute.validation.cython.experiment_cython.cfilter_dmax>`
  `cfilter_dmax <https://github.com/sdpython/teachcompute/blob/main/
  teachcompute/validation/cython/experiment_cython.pyx#L93>`_
* :func:`cfilter_dmax2 <teachcompute.validation.cython.experiment_cython.cfilter_dmax2>`
  `cfilter_dmax2 <https://github.com/sdpython/teachcompute/blob/main/
  teachcompute/validation/cython/experiment_cython.pyx#L107>`_
* :func:`cfilter_dmax4 <teachcompute.validation.cython.experiment_cython.cfilter_dmax4>`
  `cfilter_dmax4 <https://github.com/sdpython/teachcompute/blob/main/
  teachcompute/validation/cython/experiment_cython.pyx#L138>`_
* :func:`cfilter_dmax16
  <teachcompute.validation.cython.experiment_cython.cfilter_dmax16>`
  `cfilter_dmax16 <https://github.com/sdpython/teachcompute/blob/main/
  teachcompute/validation/cython/experiment_cython.pyx#L122>`_
* :func:`cyfilter_dmax <teachcompute.validation.cython.experiment_cython.cyfilter_dmax>`
  `cyfilter_dmax <https://github.com/sdpython/teachcompute/blob/main/
  teachcompute/validation/cython/experiment_cython.pyx#L72>`_
* :func:`filter_dmax_cython
  <teachcompute.validation.cython.experiment_cython.filter_dmax_cython>`
  `filter_dmax_cython <https://github.com/sdpython/teachcompute/blob/main/
  teachcompute/validation/cython/experiment_cython.pyx#L28>`_
* :func:`filter_dmax_cython_optim
  <teachcompute.validation.cython.experiment_cython.filter_dmax_cython_optim>`
  `filter_dmax_cython_optim <https://github.com/sdpython/teachcompute/blob/main/
  teachcompute/validation/cython/experiment_cython.pyx#L43>`_
* :func:`pyfilter_dmax
  <teachcompute.validation.cython.experiment_cython.pyfilter_dmax>`
  `pyfilter_dmax <https://github.com/sdpython/teachcompute/blob/main/
  teachcompute/validation/cython/experiment_cython.pyx#L15>`_

.. GENERATED FROM PYTHON SOURCE LINES 41-97

.. code-block:: Python


    import pprint
    import numpy
    import matplotlib.pyplot as plt
    from pandas import DataFrame
    from teachcompute.validation.cython.experiment_cython import (
        pyfilter_dmax,
        filter_dmax_cython,
        filter_dmax_cython_optim,
        cyfilter_dmax,
        cfilter_dmax,
        cfilter_dmax2,
        cfilter_dmax16,
        cfilter_dmax4,
    )
    from teachcompute.ext_test_case import measure_time_dim


    def get_vectors(fct, n, h=200, dtype=numpy.float64):
        ctxs = [
            dict(
                va=numpy.random.randn(n).astype(dtype),
                fil=fct,
                mx=numpy.float64(0),
                x_name=n,
            )
            for n in range(10, n, h)
        ]
        return ctxs


    def numpy_filter(va, mx):
        va[va > mx] = mx


    all_res = []
    for fct in [
        numpy_filter,
        pyfilter_dmax,
        filter_dmax_cython,
        filter_dmax_cython_optim,
        cyfilter_dmax,
        cfilter_dmax,
        cfilter_dmax2,
        cfilter_dmax16,
        cfilter_dmax4,
    ]:
        print(fct)
        ctxs = get_vectors(fct, 1000 if fct == pyfilter_dmax else 40000)
        res = list(measure_time_dim("fil(va, mx)", ctxs, verbose=1))
        for r in res:
            r["fct"] = fct.__name__
        all_res.extend(res)

    pprint.pprint(all_res[:2])





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    <function numpy_filter at 0x7ffa185cbe20>
      0%|          | 0/200 [00:00<?, ?it/s]     37%|███▋      | 74/200 [00:00<00:00, 734.22it/s]     74%|███████▍  | 148/200 [00:00<00:00, 335.10it/s]     96%|█████████▋| 193/200 [00:00<00:00, 278.68it/s]    100%|██████████| 200/200 [00:00<00:00, 282.56it/s]
    <cyfunction pyfilter_dmax at 0x7ff9972718a0>
      0%|          | 0/5 [00:00<?, ?it/s]    100%|██████████| 5/5 [00:00<00:00, 68.45it/s]
    <cyfunction filter_dmax_cython at 0x7ff9972717d0>
      0%|          | 0/200 [00:00<?, ?it/s]     54%|█████▍    | 108/200 [00:00<00:00, 1063.95it/s]    100%|██████████| 200/200 [00:00<00:00, 557.82it/s] 
    <cyfunction filter_dmax_cython_optim at 0x7ff997271700>
      0%|          | 0/200 [00:00<?, ?it/s]     53%|█████▎    | 106/200 [00:00<00:00, 1050.56it/s]    100%|██████████| 200/200 [00:00<00:00, 590.42it/s] 
    <cyfunction cyfilter_dmax at 0x7ff997271220>
      0%|          | 0/200 [00:00<?, ?it/s]     48%|████▊     | 95/200 [00:00<00:00, 945.32it/s]     95%|█████████▌| 190/200 [00:00<00:00, 565.05it/s]    100%|██████████| 200/200 [00:00<00:00, 575.55it/s]
    <cyfunction cfilter_dmax at 0x7ff997271970>
      0%|          | 0/200 [00:00<?, ?it/s]     50%|█████     | 100/200 [00:00<00:00, 996.86it/s]    100%|██████████| 200/200 [00:00<00:00, 489.97it/s]    100%|██████████| 200/200 [00:00<00:00, 530.03it/s]
    <cyfunction cfilter_dmax2 at 0x7ff997271be0>
      0%|          | 0/200 [00:00<?, ?it/s]     48%|████▊     | 96/200 [00:00<00:00, 949.86it/s]     96%|█████████▌| 191/200 [00:00<00:00, 433.61it/s]    100%|██████████| 200/200 [00:00<00:00, 448.65it/s]
    <cyfunction cfilter_dmax16 at 0x7ff997271b10>
      0%|          | 0/200 [00:00<?, ?it/s]     36%|███▌      | 71/200 [00:00<00:00, 707.81it/s]     71%|███████   | 142/200 [00:00<00:00, 365.45it/s]     94%|█████████▍| 188/200 [00:00<00:00, 267.68it/s]    100%|██████████| 200/200 [00:00<00:00, 290.62it/s]
    <cyfunction cfilter_dmax4 at 0x7ff997271a40>
      0%|          | 0/200 [00:00<?, ?it/s]     26%|██▋       | 53/200 [00:00<00:00, 528.62it/s]     53%|█████▎    | 106/200 [00:00<00:00, 238.76it/s]     69%|██████▉   | 138/200 [00:00<00:00, 169.93it/s]     80%|████████  | 161/200 [00:00<00:00, 147.09it/s]     90%|████████▉ | 179/200 [00:01<00:00, 127.18it/s]     97%|█████████▋| 194/200 [00:01<00:00, 113.76it/s]    100%|██████████| 200/200 [00:01<00:00, 143.99it/s]
    [{'average': np.float64(8.415899997089582e-07),
      'context_size': 184,
      'deviation': np.float64(1.3342516226394618e-07),
      'fct': 'numpy_filter',
      'max_exec': np.float64(1.2366200007818407e-06),
      'min_exec': np.float64(7.848800032661529e-07),
      'number': 50,
      'repeat': 10,
      'ttime': np.float64(8.415899997089581e-06),
      'warmup_time': 3.251600014664291e-05,
      'x_name': 10},
     {'average': np.float64(8.49112000196328e-07),
      'context_size': 184,
      'deviation': np.float64(8.249650193622526e-08),
      'fct': 'numpy_filter',
      'max_exec': np.float64(1.0923399986495497e-06),
      'min_exec': np.float64(8.093399992503691e-07),
      'number': 50,
      'repeat': 10,
      'ttime': np.float64(8.49112000196328e-06),
      'warmup_time': 8.533000027455273e-06,
      'x_name': 210}]




.. GENERATED FROM PYTHON SOURCE LINES 98-100

Let's display the results
+++++++++++++++++++++++++

.. GENERATED FROM PYTHON SOURCE LINES 100-120

.. code-block:: Python


    cc = DataFrame(all_res)
    cc["N"] = cc["x_name"]

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    cc[cc.N <= 1100].pivot(index="N", columns="fct", values="average").plot(
        logy=True, ax=ax[0, 0]
    )
    cc[cc.fct != "pyfilter_dmax"].pivot(index="N", columns="fct", values="average").plot(
        logy=True, ax=ax[0, 1]
    )
    cc[cc.fct != "pyfilter_dmax"].pivot(index="N", columns="fct", values="average").plot(
        logy=True, logx=True, ax=ax[1, 1]
    )
    cc[(cc.fct.str.contains("cfilter") | cc.fct.str.contains("numpy"))].pivot(
        index="N", columns="fct", values="average"
    ).plot(logy=True, ax=ax[1, 0])
    ax[0, 0].set_title("Comparison of filter implementations")
    ax[0, 1].set_title("Comparison of filter implementations\nwithout pyfilter_dmax")




.. image-sg:: /auto_examples/images/sphx_glr_plot_benchmark_filter_001.png
   :alt: Comparison of filter implementations, Comparison of filter implementations without pyfilter_dmax
   :srcset: /auto_examples/images/sphx_glr_plot_benchmark_filter_001.png
   :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 .. code-block:: none


    Text(0.5, 1.0, 'Comparison of filter implementations\nwithout pyfilter_dmax')



.. GENERATED FROM PYTHON SOURCE LINES 121-124

The results depends on the machine, its
number of cores, the compilation settings
of :epkg:`numpy` or this module.


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 5.973 seconds)


.. _sphx_glr_download_auto_examples_plot_benchmark_filter.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_benchmark_filter.ipynb <plot_benchmark_filter.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_benchmark_filter.py <plot_benchmark_filter.py>`

    .. container:: sphx-glr-download sphx-glr-download-zip

      :download:`Download zipped: plot_benchmark_filter.zip <plot_benchmark_filter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
