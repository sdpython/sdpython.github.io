
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_examples/plot_bench_cpu.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_examples_plot_bench_cpu.py>`
        to download the full example code

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_plot_bench_cpu.py:


.. _l-example-bench-cpu:

Measuring CPU performance
=========================

Processor caches must be taken into account when writing an algorithm,
see `Memory part 2: CPU caches <https://lwn.net/Articles/252125/>`_
from Ulrich Drepper.

Cache Performance
+++++++++++++++++

.. GENERATED FROM PYTHON SOURCE LINES 14-24

.. code-block:: Python

    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from pandas import DataFrame, concat
    from sphinx_runpython.runpython import run_cmd
    from onnx_extended.ext_test_case import unit_test_going
    from onnx_extended.validation.cpu._validation import (
        benchmark_cache,
        benchmark_cache_tree,
    )








.. GENERATED FROM PYTHON SOURCE LINES 25-27

Code of `benchmark_cache
<https://github.com/sdpython/onnx-extended/blob/main/onnx_extended/validation/cpu/speed_metrics.cpp#L17>`_.

.. GENERATED FROM PYTHON SOURCE LINES 27-55

.. code-block:: Python


    obs = []
    step = 2**12
    for i in tqdm(range(step, 2**20 + step, step)):
        res = min(
            [
                benchmark_cache(i, False),
                benchmark_cache(i, False),
                benchmark_cache(i, False),
            ]
        )
        if res < 0:
            # overflow
            continue
        obs.append(dict(size=i, perf=res))

    df = DataFrame(obs)
    mean = df.perf.mean()
    lag = 32
    for i in range(2, df.shape[0]):
        df.loc[i, "smooth"] = df.loc[i - 8 : i + 8, "perf"].median()
        if i > lag and i < df.shape[0] - lag:
            df.loc[i, "delta"] = (
                mean
                + df.loc[i : i + lag, "perf"].mean()
                - df.loc[i - lag + 1 : i + 1, "perf"]
            ).mean()





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

      0%|          | 0/256 [00:00<?, ?it/s]     48%|████▊     | 123/256 [00:00<00:00, 1220.72it/s]     96%|█████████▌| 246/256 [00:00<00:00, 595.68it/s]     100%|██████████| 256/256 [00:00<00:00, 620.11it/s]




.. GENERATED FROM PYTHON SOURCE LINES 56-58

Cache size estimator
++++++++++++++++++++

.. GENERATED FROM PYTHON SOURCE LINES 58-63

.. code-block:: Python


    cache_size_index = int(df.delta.argmax())
    cache_size = df.loc[cache_size_index, "size"] * 2
    print(f"L2 cache size estimation is {cache_size / 2 ** 20:1.3f} Mb.")





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    L2 cache size estimation is 0.703 Mb.




.. GENERATED FROM PYTHON SOURCE LINES 64-66

Verification
++++++++++++

.. GENERATED FROM PYTHON SOURCE LINES 66-79

.. code-block:: Python


    try:
        out, err = run_cmd("lscpu", wait=True)
        print("\n".join(_ for _ in out.split("\n") if "cache:" in _))
    except Exception as e:
        print(f"failed due to {e}")

    df = df.set_index("size")
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    df.plot(ax=ax, title="Cache Performance time/size", logy=True)
    fig.tight_layout()
    fig.savefig("plot_benchmark_cpu_array.png")




.. image-sg:: /auto_examples/images/sphx_glr_plot_bench_cpu_001.png
   :alt: Cache Performance time/size
   :srcset: /auto_examples/images/sphx_glr_plot_bench_cpu_001.png
   :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    L1d cache:                       128 KiB (4 instances)
    L1i cache:                       128 KiB (4 instances)
    L2 cache:                        1 MiB (4 instances)
    L3 cache:                        8 MiB (1 instance)




.. GENERATED FROM PYTHON SOURCE LINES 80-89

TreeEnsemble Performance
++++++++++++++++++++++++

We simulate the computation of a TreeEnsemble
of 50 features, 100 trees and depth of 10
(so :math:`2^{10}` nodes.)
The code of `benchmark_cache_tree
<https://github.com/sdpython/onnx-extended/blob/main/onnx_extended/validation/cpu/speed_metrics.cpp#L50>`_


.. GENERATED FROM PYTHON SOURCE LINES 89-117

.. code-block:: Python


    dfs = []
    cols = []
    drop = []
    for n in tqdm(range(2 if unit_test_going() else 5)):
        res = benchmark_cache_tree(
            n_rows=2000,
            n_features=50,
            n_trees=100,
            tree_size=1024,
            max_depth=10,
            search_step=64,
        )
        res = [[max(r.row, i), r.time] for i, r in enumerate(res)]
        df = DataFrame(res)
        df.columns = [f"i{n}", f"time{n}"]
        dfs.append(df)
        cols.append(df.columns[-1])
        drop.append(df.columns[0])

    df = concat(dfs, axis=1).reset_index(drop=True)
    df["i"] = df["i0"]
    df = df.drop(drop, axis=1)
    df["time_avg"] = df[cols].mean(axis=1)
    df["time_med"] = df[cols].median(axis=1)

    df.head()





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

      0%|          | 0/5 [00:00<?, ?it/s]     20%|██        | 1/5 [00:01<00:05,  1.31s/it]     40%|████      | 2/5 [00:02<00:03,  1.17s/it]     60%|██████    | 3/5 [00:03<00:02,  1.06s/it]     80%|████████  | 4/5 [00:04<00:01,  1.08s/it]    100%|██████████| 5/5 [00:05<00:00,  1.09s/it]    100%|██████████| 5/5 [00:05<00:00,  1.10s/it]


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
          <th></th>
          <th>time0</th>
          <th>time1</th>
          <th>time2</th>
          <th>time3</th>
          <th>time4</th>
          <th>i</th>
          <th>time_avg</th>
          <th>time_med</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.041219</td>
          <td>0.03867</td>
          <td>0.030491</td>
          <td>0.033571</td>
          <td>0.034151</td>
          <td>0</td>
          <td>0.035621</td>
          <td>0.034151</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.041219</td>
          <td>0.03867</td>
          <td>0.030491</td>
          <td>0.033571</td>
          <td>0.034151</td>
          <td>1</td>
          <td>0.035621</td>
          <td>0.034151</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.041219</td>
          <td>0.03867</td>
          <td>0.030491</td>
          <td>0.033571</td>
          <td>0.034151</td>
          <td>2</td>
          <td>0.035621</td>
          <td>0.034151</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.041219</td>
          <td>0.03867</td>
          <td>0.030491</td>
          <td>0.033571</td>
          <td>0.034151</td>
          <td>3</td>
          <td>0.035621</td>
          <td>0.034151</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.041219</td>
          <td>0.03867</td>
          <td>0.030491</td>
          <td>0.033571</td>
          <td>0.034151</td>
          <td>4</td>
          <td>0.035621</td>
          <td>0.034151</td>
        </tr>
      </tbody>
    </table>
    </div>
    </div>
    <br />
    <br />

.. GENERATED FROM PYTHON SOURCE LINES 118-120

Estimation
++++++++++

.. GENERATED FROM PYTHON SOURCE LINES 120-128

.. code-block:: Python


    print("Optimal batch size is among:")
    dfi = df[["time_med", "i"]].groupby("time_med").min()
    dfi_min = set(dfi["i"])
    dfsub = df[df["i"].isin(dfi_min)]
    dfs = dfsub.sort_values("time_med").reset_index()
    print(dfs[["i", "time_med", "time_avg"]].head(10))





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    Optimal batch size is among:
          i  time_med  time_avg
    0  1280  0.032751  0.033592
    1  1600  0.032936  0.033915
    2  1408  0.032959  0.033938
    3  1344  0.033052  0.034085
    4  1216  0.033169  0.034291
    5  1536  0.033426  0.033785
    6  1472  0.033493  0.033620
    7  1152  0.033496  0.034325
    8  1664  0.033597  0.034121
    9  1792  0.033612  0.036147




.. GENERATED FROM PYTHON SOURCE LINES 129-130

One possible estimation

.. GENERATED FROM PYTHON SOURCE LINES 130-135

.. code-block:: Python


    subdfs = dfs[:20]
    avg = (subdfs["i"] / subdfs["time_avg"]).sum() / (subdfs["time_avg"] ** (-1)).sum()
    print(f"Estimation: {avg}")





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    Estimation: 1202.786800709671




.. GENERATED FROM PYTHON SOURCE LINES 136-137

Plots.

.. GENERATED FROM PYTHON SOURCE LINES 137-146

.. code-block:: Python


    cols_time = ["time_avg", "time_med"]
    fig, ax = plt.subplots(2, 1, figsize=(12, 6))
    df.set_index("i").drop(cols_time, axis=1).plot(
        ax=ax[0], title="TreeEnsemble Performance time per row", logy=True, linewidth=0.2
    )
    df.set_index("i")[cols_time].plot(ax=ax[1], linewidth=1.0, logy=True)
    fig.tight_layout()
    fig.savefig("plot_bench_cpu.png")



.. image-sg:: /auto_examples/images/sphx_glr_plot_bench_cpu_002.png
   :alt: TreeEnsemble Performance time per row
   :srcset: /auto_examples/images/sphx_glr_plot_bench_cpu_002.png
   :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 7.694 seconds)


.. _sphx_glr_download_auto_examples_plot_bench_cpu.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_bench_cpu.ipynb <plot_bench_cpu.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_bench_cpu.py <plot_bench_cpu.py>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
