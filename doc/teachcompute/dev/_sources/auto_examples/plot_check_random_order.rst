
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_examples/plot_check_random_order.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_examples_plot_check_random_order.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_plot_check_random_order.py:


Random order for a sum
======================

Parallelization usually means a summation is done with a random order.
That may lead to different values if the computation is made many times
even though the result should be the same. This example compares
summation of random permutation of the same array of values.

Setup
+++++

.. GENERATED FROM PYTHON SOURCE LINES 13-33

.. code-block:: Python


    from tqdm import tqdm
    import numpy as np
    import pandas

    unique_values = np.array(
        [2.1102535724639893, 0.5986238718032837, -0.49545818567276], dtype=np.float32
    )
    random_index = np.random.randint(0, 3, 2000)
    assert set(random_index) == {0, 1, 2}
    values = unique_values[random_index]

    s0 = values.sum()
    s1 = np.array(0, dtype=np.float32)
    for n in values:
        s1 += n

    delta = s1 - s0
    print(f"reduced sum={s0}, iterative sum={s1}, delta={delta}")





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    reduced sum=1533.336669921875, iterative sum=1533.3216552734375, delta=-0.0150146484375




.. GENERATED FROM PYTHON SOURCE LINES 34-41

There are discrepancies.

Random order
++++++++++++

Let's go further and check the sum of random permutation of the same set.
Let's compare the result with the same sum done with a higher precision (double).

.. GENERATED FROM PYTHON SOURCE LINES 41-78

.. code-block:: Python



    def check_orders(values, n=200, bias=0):
        double_sums = []
        sums = []
        reduced_sums = []
        reduced_dsums = []
        for _i in tqdm(range(n)):
            permuted_values = np.random.permutation(values)
            s = np.array(bias, dtype=np.float32)
            sd = np.array(bias, dtype=np.float64)
            for n in permuted_values:
                s += n
                sd += n
            sums.append(s)
            double_sums.append(sd)
            reduced_sums.append(permuted_values.sum() + bias)
            reduced_dsums.append(permuted_values.astype(np.float64).sum() + bias)

        data = []
        mi, ma = min(sums), max(sums)
        data.append(dict(name="seq_fp32", min=mi, max=ma, bias=bias))
        print(f"min={mi} max={ma} delta={ma-mi}")
        mi, ma = min(double_sums), max(double_sums)
        data.append(dict(name="seq_fp64", min=mi, max=ma, bias=bias))
        print(f"min={mi} max={ma} delta={ma-mi} (double)")
        mi, ma = min(reduced_sums), max(reduced_sums)
        data.append(dict(name="red_f32", min=mi, max=ma, bias=bias))
        print(f"min={mi} max={ma} delta={ma-mi} (reduced)")
        mi, ma = min(reduced_dsums), max(reduced_dsums)
        data.append(dict(name="red_f64", min=mi, max=ma, bias=bias))
        print(f"min={mi} max={ma} delta={ma-mi} (reduced)")
        return data


    data1 = check_orders(values)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

      0%|          | 0/200 [00:00<?, ?it/s]      8%|▊         | 17/200 [00:00<00:01, 163.06it/s]     17%|█▋        | 34/200 [00:00<00:01, 129.65it/s]     28%|██▊       | 56/200 [00:00<00:00, 163.30it/s]     40%|███▉      | 79/200 [00:00<00:00, 184.17it/s]     50%|█████     | 101/200 [00:00<00:00, 193.45it/s]     62%|██████▏   | 123/200 [00:00<00:00, 200.58it/s]     72%|███████▎  | 145/200 [00:00<00:00, 200.43it/s]     83%|████████▎ | 166/200 [00:00<00:00, 196.52it/s]     94%|█████████▍| 189/200 [00:00<00:00, 204.47it/s]    100%|██████████| 200/200 [00:01<00:00, 191.60it/s]
    min=1533.320068359375 max=1533.3232421875 delta=0.003173828125
    min=1533.336667060852 max=1533.336667060852 delta=0.0 (double)
    min=1533.3365478515625 max=1533.336669921875 delta=0.0001220703125 (reduced)
    min=1533.336667060852 max=1533.336667060852 delta=0.0 (reduced)




.. GENERATED FROM PYTHON SOURCE LINES 79-91

This example clearly shows the order has an impact.
It is usually unavoidable but it could reduced if the sum
it close to zero. In that case, the sum would be of the same
order of magnitude of the add values.

Removing the average
++++++++++++++++++++

Computing the average of the values requires to compute the sum.
However if we have an estimator of this average, not necessarily
the exact value, we would help the summation to keep the same order
of magnitude than the values it adds.

.. GENERATED FROM PYTHON SOURCE LINES 91-96

.. code-block:: Python


    mean = unique_values.mean()
    values -= mean
    data2 = check_orders(values, bias=len(values) * mean)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

      0%|          | 0/200 [00:00<?, ?it/s]     11%|█         | 22/200 [00:00<00:00, 218.63it/s]     22%|██▏       | 44/200 [00:00<00:00, 206.39it/s]     32%|███▎      | 65/200 [00:00<00:00, 202.67it/s]     44%|████▎     | 87/200 [00:00<00:00, 205.66it/s]     54%|█████▍    | 108/200 [00:00<00:00, 207.04it/s]     64%|██████▍   | 129/200 [00:00<00:00, 202.51it/s]     75%|███████▌  | 150/200 [00:00<00:00, 187.33it/s]     86%|████████▌ | 171/200 [00:00<00:00, 191.26it/s]     96%|█████████▋| 193/200 [00:00<00:00, 198.71it/s]    100%|██████████| 200/200 [00:01<00:00, 199.94it/s]
    min=1533.3370361328125 max=1533.3370361328125 delta=0.0
    min=1533.336665213108 max=1533.336665213108 delta=0.0 (double)
    min=1533.336669921875 max=1533.336669921875 delta=0.0 (reduced)
    min=1533.336665213108 max=1533.336665213108 delta=0.0 (reduced)




.. GENERATED FROM PYTHON SOURCE LINES 97-98

The differences are clearly lower.

.. GENERATED FROM PYTHON SOURCE LINES 98-104

.. code-block:: Python


    df = pandas.DataFrame(data1 + data2)
    df["delta"] = df["max"] - df["min"]
    piv = df.pivot(index="name", columns="bias", values="delta")
    print(piv)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    bias     0.000000    1475.613037
    name                            
    red_f32     0.000122         0.0
    red_f64          0.0         0.0
    seq_fp32    0.003174         0.0
    seq_fp64         0.0         0.0




.. GENERATED FROM PYTHON SOURCE LINES 105-106

Plots.

.. GENERATED FROM PYTHON SOURCE LINES 106-111

.. code-block:: Python


    ax = piv.plot.barh()
    ax.set_title("max(sum) - min(sum) over random orders")
    ax.get_figure().tight_layout()
    ax.get_figure().savefig("plot_check_random_order.png")



.. image-sg:: /auto_examples/images/sphx_glr_plot_check_random_order_001.png
   :alt: max(sum) - min(sum) over random orders
   :srcset: /auto_examples/images/sphx_glr_plot_check_random_order_001.png
   :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 2.220 seconds)


.. _sphx_glr_download_auto_examples_plot_check_random_order.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_check_random_order.ipynb <plot_check_random_order.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_check_random_order.py <plot_check_random_order.py>`

    .. container:: sphx-glr-download sphx-glr-download-zip

      :download:`Download zipped: plot_check_random_order.zip <plot_check_random_order.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
