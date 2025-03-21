
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_examples/plot_bench_cypy_ort.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_examples_plot_bench_cypy_ort.py>`
        to download the full example code

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

    t_ort={'average': 1.1017984999807592e-05, 'deviation': 1.1627572347791087e-06, 'min_exec': 1.0173499999837077e-05, 'max_exec': 1.705549999996947e-05, 'repeat': 100, 'number': 200, 'ttime': 0.0011017984999807592, 'context_size': 64, 'warmup_time': 6.429999984902679e-05}
    t_ext={'average': 1.4125910000029763e-05, 'deviation': 1.0116295268419106e-05, 'min_exec': 9.202999999615713e-06, 'max_exec': 7.462750000058804e-05, 'repeat': 100, 'number': 200, 'ttime': 0.0014125910000029763, 'context_size': 64, 'warmup_time': 7.379999988188501e-05}
    t_ext2={'average': 8.863004999875556e-06, 'deviation': 1.1311618198577438e-06, 'min_exec': 7.898999999724765e-06, 'max_exec': 1.5066000000842906e-05, 'repeat': 100, 'number': 200, 'ttime': 0.0008863004999875556, 'context_size': 64, 'warmup_time': 4.759999956149841e-05}




.. GENERATED FROM PYTHON SOURCE LINES 90-92

Benchmark
+++++++++

.. GENERATED FROM PYTHON SOURCE LINES 92-125

.. code-block:: Python

    dims = list(int(i) for i in script_args.dims.split(","))

    data = []
    for dim in tqdm(dims):
        if dim < 1000:
            number, repeat = script_args.number, script_args.repeat
        else:
            number, repeat = script_args.number * 5, script_args.repeat * 5
        x = numpy.random.randn(dim, dim).astype(numpy.float32)
        t_ort = measure_time(
            lambda: sess_ort.run(None, {"X": x})[0], number=number, repeat=50
        )
        t_ort["name"] = "ort"
        t_ort["dim"] = dim
        data.append(t_ort)

        t_ext = measure_time(lambda: sess_ext.run([x])[0], number=number, repeat=repeat)
        t_ext["name"] = "ext"
        t_ext["dim"] = dim
        data.append(t_ext)

        t_ext2 = measure_time(lambda: sess_ext.run_1_1(x), number=number, repeat=repeat)
        t_ext2["name"] = "ext_1_1"
        t_ext2["dim"] = dim
        data.append(t_ext2)

        if unit_test_going() and dim >= 10:
            break


    df = DataFrame(data)
    df





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

      0%|          | 0/4 [00:00<?, ?it/s]    100%|██████████| 4/4 [00:03<00:00,  1.12it/s]    100%|██████████| 4/4 [00:03<00:00,  1.12it/s]


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
          <td>0.000011</td>
          <td>6.732164e-07</td>
          <td>0.000010</td>
          <td>0.000015</td>
          <td>50</td>
          <td>10</td>
          <td>0.000539</td>
          <td>64</td>
          <td>0.000090</td>
          <td>ort</td>
          <td>1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.000009</td>
          <td>1.841304e-07</td>
          <td>0.000009</td>
          <td>0.000010</td>
          <td>10</td>
          <td>10</td>
          <td>0.000094</td>
          <td>64</td>
          <td>0.000040</td>
          <td>ext</td>
          <td>1</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.000008</td>
          <td>8.129576e-08</td>
          <td>0.000008</td>
          <td>0.000008</td>
          <td>10</td>
          <td>10</td>
          <td>0.000079</td>
          <td>64</td>
          <td>0.000022</td>
          <td>ext_1_1</td>
          <td>1</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.000011</td>
          <td>5.452700e-07</td>
          <td>0.000011</td>
          <td>0.000015</td>
          <td>50</td>
          <td>10</td>
          <td>0.000546</td>
          <td>64</td>
          <td>0.000032</td>
          <td>ort</td>
          <td>10</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.000010</td>
          <td>7.448463e-07</td>
          <td>0.000009</td>
          <td>0.000012</td>
          <td>10</td>
          <td>10</td>
          <td>0.000096</td>
          <td>64</td>
          <td>0.000029</td>
          <td>ext</td>
          <td>10</td>
        </tr>
        <tr>
          <th>5</th>
          <td>0.000008</td>
          <td>9.436630e-08</td>
          <td>0.000008</td>
          <td>0.000008</td>
          <td>10</td>
          <td>10</td>
          <td>0.000080</td>
          <td>64</td>
          <td>0.000021</td>
          <td>ext_1_1</td>
          <td>10</td>
        </tr>
        <tr>
          <th>6</th>
          <td>0.000015</td>
          <td>3.044421e-06</td>
          <td>0.000014</td>
          <td>0.000033</td>
          <td>50</td>
          <td>10</td>
          <td>0.000748</td>
          <td>64</td>
          <td>0.000036</td>
          <td>ort</td>
          <td>100</td>
        </tr>
        <tr>
          <th>7</th>
          <td>0.000014</td>
          <td>2.304404e-06</td>
          <td>0.000013</td>
          <td>0.000019</td>
          <td>10</td>
          <td>10</td>
          <td>0.000139</td>
          <td>64</td>
          <td>0.000057</td>
          <td>ext</td>
          <td>100</td>
        </tr>
        <tr>
          <th>8</th>
          <td>0.000011</td>
          <td>9.202717e-08</td>
          <td>0.000011</td>
          <td>0.000011</td>
          <td>10</td>
          <td>10</td>
          <td>0.000112</td>
          <td>64</td>
          <td>0.000025</td>
          <td>ext_1_1</td>
          <td>100</td>
        </tr>
        <tr>
          <th>9</th>
          <td>0.000485</td>
          <td>1.299964e-04</td>
          <td>0.000371</td>
          <td>0.001006</td>
          <td>50</td>
          <td>50</td>
          <td>0.024248</td>
          <td>64</td>
          <td>0.001992</td>
          <td>ort</td>
          <td>1000</td>
        </tr>
        <tr>
          <th>10</th>
          <td>0.000462</td>
          <td>8.708347e-05</td>
          <td>0.000365</td>
          <td>0.000741</td>
          <td>50</td>
          <td>50</td>
          <td>0.023108</td>
          <td>64</td>
          <td>0.002587</td>
          <td>ext</td>
          <td>1000</td>
        </tr>
        <tr>
          <th>11</th>
          <td>0.000454</td>
          <td>5.599289e-05</td>
          <td>0.000372</td>
          <td>0.000642</td>
          <td>50</td>
          <td>50</td>
          <td>0.022699</td>
          <td>64</td>
          <td>0.000430</td>
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

   **Total running time of the script:** (0 minutes 4.796 seconds)


.. _sphx_glr_download_auto_examples_plot_bench_cypy_ort.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_bench_cypy_ort.ipynb <plot_bench_cypy_ort.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_bench_cypy_ort.py <plot_bench_cypy_ort.py>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
