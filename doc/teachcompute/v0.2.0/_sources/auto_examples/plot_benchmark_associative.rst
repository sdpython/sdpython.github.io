
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_examples/plot_benchmark_associative.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_examples_plot_benchmark_associative.py>`
        to download the full example code

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_plot_benchmark_associative.py:


Associativity and matrix multiplication
=======================================

The matrix multiplication `m1 @ m2 @ m3` can be done
in two different ways: `(m1 @ m2) @ m3` or `m1 @ (m2 @ m3)`.
Are these two orders equivalent or is there a better order?

.. GENERATED FROM PYTHON SOURCE LINES 9-17

.. code-block:: Python


    import pprint
    import numpy
    import matplotlib.pyplot as plt
    from pandas import DataFrame
    from tqdm import tqdm
    from teachcompute.ext_test_case import measure_time








.. GENERATED FROM PYTHON SOURCE LINES 18-21

First try
+++++++++


.. GENERATED FROM PYTHON SOURCE LINES 21-41

.. code-block:: Python


    m1 = numpy.random.rand(100, 100)
    m2 = numpy.random.rand(100, 10)
    m3 = numpy.random.rand(10, 100)

    m = m1 @ m2 @ m3

    print(m.shape)

    mm1 = (m1 @ m2) @ m3
    mm2 = m1 @ (m2 @ m3)

    print(mm1.shape, mm2.shape)

    t1 = measure_time(lambda: (m1 @ m2) @ m3, context={}, number=50, repeat=50)
    pprint.pprint(t1)

    t2 = measure_time(lambda: m1 @ (m2 @ m3), context={}, number=50, repeat=50)
    pprint.pprint(t2)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    (100, 100)
    (100, 100) (100, 100)
    {'average': 7.918572000035058e-05,
     'context_size': 64,
     'deviation': 7.472517220805028e-05,
     'max_exec': 0.0002890740000020742,
     'min_exec': 2.662199999576842e-05,
     'number': 50,
     'repeat': 50,
     'ttime': 0.003959286000017529,
     'warmup_time': 0.00012929999957123073}
    {'average': 0.003669156359999761,
     'context_size': 64,
     'deviation': 0.0058324362524603416,
     'max_exec': 0.021446986000000834,
     'min_exec': 7.507799999984854e-05,
     'number': 50,
     'repeat': 50,
     'ttime': 0.18345781799998806,
     'warmup_time': 0.025857000000087282}




.. GENERATED FROM PYTHON SOURCE LINES 42-44

With different sizes
++++++++++++++++++++

.. GENERATED FROM PYTHON SOURCE LINES 44-64

.. code-block:: Python


    obs = []
    for i in tqdm([50, 100, 125, 150, 175, 200]):
        m1 = numpy.random.rand(i, i)
        m2 = numpy.random.rand(i, 10)
        m3 = numpy.random.rand(10, i)

        t1 = measure_time(lambda: (m1 @ m2) @ m3, context={}, number=50, repeat=50)
        t1["formula"] = "(m1 @ m2) @ m3"
        t1["size"] = i
        obs.append(t1)
        t2 = measure_time(lambda: m1 @ (m2 @ m3), context={}, number=50, repeat=50)
        t2["formula"] = "m1 @ (m2 @ m3)"
        t2["size"] = i
        obs.append(t2)

    df = DataFrame(obs)
    piv = df.pivot(index="size", columns="formula", values="average")
    piv





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

      0%|          | 0/6 [00:00<?, ?it/s]     17%|█▋        | 1/6 [00:00<00:00,  5.72it/s]     33%|███▎      | 2/6 [00:02<00:04,  1.16s/it]     50%|█████     | 3/6 [00:02<00:02,  1.12it/s]     67%|██████▋   | 4/6 [00:04<00:02,  1.10s/it]     83%|████████▎ | 5/6 [00:04<00:01,  1.04s/it]    100%|██████████| 6/6 [00:06<00:00,  1.26s/it]    100%|██████████| 6/6 [00:06<00:00,  1.11s/it]


.. raw:: html

    <div class="output_subarea output_html rendered_html output_result">
    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th>formula</th>
          <th>(m1 @ m2) @ m3</th>
          <th>m1 @ (m2 @ m3)</th>
        </tr>
        <tr>
          <th>size</th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>50</th>
          <td>0.000027</td>
          <td>0.000041</td>
        </tr>
        <tr>
          <th>100</th>
          <td>0.000040</td>
          <td>0.000692</td>
        </tr>
        <tr>
          <th>125</th>
          <td>0.000070</td>
          <td>0.000160</td>
        </tr>
        <tr>
          <th>150</th>
          <td>0.000081</td>
          <td>0.000488</td>
        </tr>
        <tr>
          <th>175</th>
          <td>0.000086</td>
          <td>0.000285</td>
        </tr>
        <tr>
          <th>200</th>
          <td>0.000131</td>
          <td>0.000540</td>
        </tr>
      </tbody>
    </table>
    </div>
    </div>
    <br />
    <br />

.. GENERATED FROM PYTHON SOURCE LINES 65-67

Graph
+++++

.. GENERATED FROM PYTHON SOURCE LINES 67-77

.. code-block:: Python


    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    piv.plot(
        logx=True,
        logy=True,
        ax=ax[0],
        title=f"{m1.shape!r} @ {m2.shape!r} @ " f"{m3.shape!r}".replace("200", "size"),
    )
    piv["ratio"] = piv["m1 @ (m2 @ m3)"] / piv["(m1 @ m2) @ m3"]
    piv[["ratio"]].plot(ax=ax[1])



.. image-sg:: /auto_examples/images/sphx_glr_plot_benchmark_associative_001.png
   :alt: (size, size) @ (size, 10) @ (10, size)
   :srcset: /auto_examples/images/sphx_glr_plot_benchmark_associative_001.png
   :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 .. code-block:: none


    <Axes: xlabel='size'>




.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 16.443 seconds)


.. _sphx_glr_download_auto_examples_plot_benchmark_associative.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_benchmark_associative.ipynb <plot_benchmark_associative.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_benchmark_associative.py <plot_benchmark_associative.py>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
