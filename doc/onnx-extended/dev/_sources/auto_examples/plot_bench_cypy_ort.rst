
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_examples/plot_bench_cypy_ort.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_examples_plot_bench_cypy_ort.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_plot_bench_cypy_ort.py:


.. _l-cython-pybind11-ort-bindings:

Measuring onnxruntime performance against a cython binding
==========================================================

The following code measures the performance of the python bindings
against a :epkg:`cython` binding.
The time spent in it is not significant when the computation is huge
but it may be for small matrices.

.. GENERATED FROM PYTHON SOURCE LINES 12-42

.. code-block:: Python


    import numpy
    from pandas import DataFrame
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from onnx import numpy_helper, TensorProto
    from onnx.helper import (
        make_model,
        make_node,
        make_graph,
        make_tensor_value_info,
        make_opsetid,
    )
    from onnx.checker import check_model
    from onnxruntime import InferenceSession
    from onnx_extended.ortcy.wrap.ortinf import OrtSession
    from onnx_extended.args import get_parsed_args
    from onnx_extended.ext_test_case import measure_time, unit_test_going


    script_args = get_parsed_args(
        "plot_bench_cypy_ort",
        description=__doc__,
        dims=(
            "1,10" if unit_test_going() else "1,10,100,1000",
            "square matrix dimensions to try, comma separated values",
        ),
        expose="repeat,number",
    )








.. GENERATED FROM PYTHON SOURCE LINES 43-45

A simple onnx model
+++++++++++++++++++

.. GENERATED FROM PYTHON SOURCE LINES 45-55

.. code-block:: Python



    A = numpy_helper.from_array(numpy.array([1], dtype=numpy.float32), name="A")
    X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
    Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None, None])
    node1 = make_node("Add", ["X", "A"], ["Y"])
    graph = make_graph([node1], "+1", [X], [Y], [A])
    onnx_model = make_model(graph, opset_imports=[make_opsetid("", 18)], ir_version=8)
    check_model(onnx_model)








.. GENERATED FROM PYTHON SOURCE LINES 56-58

Two python bindings on CPU
++++++++++++++++++++++++++

.. GENERATED FROM PYTHON SOURCE LINES 58-74

.. code-block:: Python


    sess_ort = InferenceSession(
        onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    sess_ext = OrtSession(onnx_model.SerializeToString())

    x = numpy.random.randn(10, 10).astype(numpy.float32)
    y = x + 1

    y_ort = sess_ort.run(None, {"X": x})[0]
    y_ext = sess_ext.run([x])[0]

    d_ort = numpy.abs(y_ort - y).sum()
    d_ext = numpy.abs(y_ext - y).sum()
    print(f"Discrepancies: d_ort={d_ort}, d_ext={d_ext}")





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    Discrepancies: d_ort=0.0, d_ext=0.0




.. GENERATED FROM PYTHON SOURCE LINES 75-79

Time measurement
++++++++++++++++

*run_1_1* is a specific implementation when there is only 1 input and output.

.. GENERATED FROM PYTHON SOURCE LINES 79-89

.. code-block:: Python


    t_ort = measure_time(lambda: sess_ort.run(None, {"X": x})[0], number=200, repeat=100)
    print(f"t_ort={t_ort}")

    t_ext = measure_time(lambda: sess_ext.run([x])[0], number=200, repeat=100)
    print(f"t_ext={t_ext}")

    t_ext2 = measure_time(lambda: sess_ext.run_1_1(x), number=200, repeat=100)
    print(f"t_ext2={t_ext2}")





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    t_ort={'average': np.float64(5.4070025496912415e-06), 'deviation': np.float64(2.1811158847588845e-06), 'min_exec': np.float64(4.663300005631754e-06), 'max_exec': np.float64(2.5058569999600877e-05), 'repeat': 100, 'number': 200, 'ttime': np.float64(0.0005407002549691241), 'context_size': 64, 'warmup_time': 0.0001342309988103807}
    t_ext={'average': np.float64(5.1815271996019875e-06), 'deviation': np.float64(4.304988726654559e-07), 'min_exec': np.float64(4.901085003439221e-06), 'max_exec': np.float64(7.735379995210678e-06), 'repeat': 100, 'number': 200, 'ttime': np.float64(0.0005181527199601988), 'context_size': 64, 'warmup_time': 8.35049995657755e-05}
    t_ext2={'average': np.float64(4.72687039982702e-06), 'deviation': np.float64(5.308661812130688e-07), 'min_exec': np.float64(4.307794997657766e-06), 'max_exec': np.float64(7.513645005019498e-06), 'repeat': 100, 'number': 200, 'ttime': np.float64(0.00047268703998270205), 'context_size': 64, 'warmup_time': 2.5773000743356533e-05}




.. GENERATED FROM PYTHON SOURCE LINES 90-92

Benchmark
+++++++++

.. GENERATED FROM PYTHON SOURCE LINES 92-125

.. code-block:: Python

    dims = [int(i) for i in script_args.dims.split(",")]

    data = []
    for dim in tqdm(dims):
        if dim < 1000:
            number, repeat = script_args.number, script_args.repeat
        else:
            number, repeat = script_args.number * 5, script_args.repeat * 5
        x = numpy.random.randn(dim, dim).astype(numpy.float32)
        t_ort = measure_time(
            lambda x=x: sess_ort.run(None, {"X": x})[0], number=number, repeat=50
        )
        t_ort["name"] = "ort"
        t_ort["dim"] = dim
        data.append(t_ort)

        t_ext = measure_time(lambda x=x: sess_ext.run([x])[0], number=number, repeat=repeat)
        t_ext["name"] = "ext"
        t_ext["dim"] = dim
        data.append(t_ext)

        t_ext2 = measure_time(lambda x=x: sess_ext.run_1_1(x), number=number, repeat=repeat)
        t_ext2["name"] = "ext_1_1"
        t_ext2["dim"] = dim
        data.append(t_ext2)

        if unit_test_going() and dim >= 10:
            break


    df = DataFrame(data)
    df





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

      0%|          | 0/4 [00:00<?, ?it/s]    100%|██████████| 4/4 [00:01<00:00,  2.38it/s]    100%|██████████| 4/4 [00:01<00:00,  2.38it/s]


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
          <th>average</th>
          <th>deviation</th>
          <th>min_exec</th>
          <th>max_exec</th>
          <th>repeat</th>
          <th>number</th>
          <th>ttime</th>
          <th>context_size</th>
          <th>warmup_time</th>
          <th>name</th>
          <th>dim</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.000005</td>
          <td>3.113291e-07</td>
          <td>0.000005</td>
          <td>0.000007</td>
          <td>50</td>
          <td>10</td>
          <td>0.000254</td>
          <td>64</td>
          <td>0.000072</td>
          <td>ort</td>
          <td>1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.000006</td>
          <td>1.095015e-06</td>
          <td>0.000005</td>
          <td>0.000009</td>
          <td>10</td>
          <td>10</td>
          <td>0.000056</td>
          <td>64</td>
          <td>0.000068</td>
          <td>ext</td>
          <td>1</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.000007</td>
          <td>4.366461e-06</td>
          <td>0.000004</td>
          <td>0.000018</td>
          <td>10</td>
          <td>10</td>
          <td>0.000071</td>
          <td>64</td>
          <td>0.000016</td>
          <td>ext_1_1</td>
          <td>1</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.000006</td>
          <td>3.298409e-06</td>
          <td>0.000005</td>
          <td>0.000022</td>
          <td>50</td>
          <td>10</td>
          <td>0.000288</td>
          <td>64</td>
          <td>0.000042</td>
          <td>ort</td>
          <td>10</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.000005</td>
          <td>3.370656e-07</td>
          <td>0.000005</td>
          <td>0.000006</td>
          <td>10</td>
          <td>10</td>
          <td>0.000053</td>
          <td>64</td>
          <td>0.000059</td>
          <td>ext</td>
          <td>10</td>
        </tr>
        <tr>
          <th>5</th>
          <td>0.000006</td>
          <td>5.555920e-06</td>
          <td>0.000004</td>
          <td>0.000023</td>
          <td>10</td>
          <td>10</td>
          <td>0.000064</td>
          <td>64</td>
          <td>0.000015</td>
          <td>ext_1_1</td>
          <td>10</td>
        </tr>
        <tr>
          <th>6</th>
          <td>0.000006</td>
          <td>1.399302e-06</td>
          <td>0.000006</td>
          <td>0.000013</td>
          <td>50</td>
          <td>10</td>
          <td>0.000313</td>
          <td>64</td>
          <td>0.000038</td>
          <td>ort</td>
          <td>100</td>
        </tr>
        <tr>
          <th>7</th>
          <td>0.000007</td>
          <td>2.568428e-07</td>
          <td>0.000007</td>
          <td>0.000008</td>
          <td>10</td>
          <td>10</td>
          <td>0.000071</td>
          <td>64</td>
          <td>0.000042</td>
          <td>ext</td>
          <td>100</td>
        </tr>
        <tr>
          <th>8</th>
          <td>0.000006</td>
          <td>7.985796e-08</td>
          <td>0.000006</td>
          <td>0.000007</td>
          <td>10</td>
          <td>10</td>
          <td>0.000064</td>
          <td>64</td>
          <td>0.000016</td>
          <td>ext_1_1</td>
          <td>100</td>
        </tr>
        <tr>
          <th>9</th>
          <td>0.000062</td>
          <td>2.124362e-05</td>
          <td>0.000035</td>
          <td>0.000132</td>
          <td>50</td>
          <td>50</td>
          <td>0.003124</td>
          <td>64</td>
          <td>0.005340</td>
          <td>ort</td>
          <td>1000</td>
        </tr>
        <tr>
          <th>10</th>
          <td>0.000294</td>
          <td>5.240357e-05</td>
          <td>0.000247</td>
          <td>0.000469</td>
          <td>50</td>
          <td>50</td>
          <td>0.014722</td>
          <td>64</td>
          <td>0.003543</td>
          <td>ext</td>
          <td>1000</td>
        </tr>
        <tr>
          <th>11</th>
          <td>0.000297</td>
          <td>7.914858e-05</td>
          <td>0.000241</td>
          <td>0.000518</td>
          <td>50</td>
          <td>50</td>
          <td>0.014847</td>
          <td>64</td>
          <td>0.000593</td>
          <td>ext_1_1</td>
          <td>1000</td>
        </tr>
      </tbody>
    </table>
    </div>
    </div>
    <br />
    <br />

.. GENERATED FROM PYTHON SOURCE LINES 126-128

Plots
+++++

.. GENERATED FROM PYTHON SOURCE LINES 128-135

.. code-block:: Python


    piv = df.pivot(index="dim", columns="name", values="average")

    fig, ax = plt.subplots(1, 1)
    piv.plot(ax=ax, title="Binding Comparison", logy=True, logx=True)
    fig.tight_layout()
    fig.savefig("plot_bench_ort.png")



.. image-sg:: /auto_examples/images/sphx_glr_plot_bench_cypy_ort_001.png
   :alt: Binding Comparison
   :srcset: /auto_examples/images/sphx_glr_plot_bench_cypy_ort_001.png
   :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 2.464 seconds)


.. _sphx_glr_download_auto_examples_plot_bench_cypy_ort.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_bench_cypy_ort.ipynb <plot_bench_cypy_ort.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_bench_cypy_ort.py <plot_bench_cypy_ort.py>`

    .. container:: sphx-glr-download sphx-glr-download-zip

      :download:`Download zipped: plot_bench_cypy_ort.zip <plot_bench_cypy_ort.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
