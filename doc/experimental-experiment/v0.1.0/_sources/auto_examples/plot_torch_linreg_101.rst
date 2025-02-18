
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_examples/plot_torch_linreg_101.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_examples_plot_torch_linreg_101.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_plot_torch_linreg_101.py:


.. _l-plot-torch-linreg-101:

=========================================
101: Linear Regression and export to ONNX
=========================================

:epkg:`scikit-learn` and :epkg:`torch` to train a linear regression.

data
====

.. GENERATED FROM PYTHON SOURCE LINES 13-31

.. code-block:: Python


    import numpy as np
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression, SGDRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    import torch
    from onnxruntime import InferenceSession
    from experimental_experiment.helpers import pretty_onnx
    from onnx_array_api.plotting.graphviz_helper import plot_dot
    from experimental_experiment.torch_interpreter import to_onnx


    X, y = make_regression(1000, n_features=5, noise=10.0, n_informative=2)
    print(X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    (1000, 5) (1000,)




.. GENERATED FROM PYTHON SOURCE LINES 32-38

scikit-learn: the simple regression
===================================

.. math::

      A^* = (X'X)^{-1}X'Y

.. GENERATED FROM PYTHON SOURCE LINES 38-45

.. code-block:: Python



    clr = LinearRegression()
    clr.fit(X_train, y_train)

    print(f"coefficients: {clr.coef_}, {clr.intercept_}")





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    coefficients: [ 1.49148391e-01  9.55477138e+01  4.27602676e+01 -5.16603994e-01
     -9.13095100e-02], 0.15098552078709382




.. GENERATED FROM PYTHON SOURCE LINES 46-48

Evaluation
==========

.. GENERATED FROM PYTHON SOURCE LINES 48-54

.. code-block:: Python


    y_pred = clr.predict(X_test)
    l2 = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"LinearRegression: l2={l2}, r2={r2}")





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    LinearRegression: l2=101.08471574277934, r2=0.9895516762682767




.. GENERATED FROM PYTHON SOURCE LINES 55-59

scikit-learn: SGD algorithm
===================================

SGD = Stochastic Gradient Descent

.. GENERATED FROM PYTHON SOURCE LINES 59-65

.. code-block:: Python


    clr = SGDRegressor(max_iter=5, verbose=1)
    clr.fit(X_train, y_train)

    print(f"coefficients: {clr.coef_}, {clr.intercept_}")





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    -- Epoch 1
    Norm: 89.74, NNZs: 5, Bias: -3.207615, T: 750, Avg. loss: 1192.340618
    Total training time: 0.00 seconds.
    -- Epoch 2
    Norm: 100.77, NNZs: 5, Bias: -1.122405, T: 1500, Avg. loss: 93.318303
    Total training time: 0.00 seconds.
    -- Epoch 3
    Norm: 103.38, NNZs: 5, Bias: -0.307030, T: 2250, Avg. loss: 55.821301
    Total training time: 0.00 seconds.
    -- Epoch 4
    Norm: 104.26, NNZs: 5, Bias: -0.081255, T: 3000, Avg. loss: 52.718419
    Total training time: 0.00 seconds.
    -- Epoch 5
    Norm: 104.47, NNZs: 5, Bias: 0.059221, T: 3750, Avg. loss: 52.405686
    Total training time: 0.00 seconds.
    /home/xadupre/vv/this312/lib/python3.12/site-packages/sklearn/linear_model/_stochastic_gradient.py:1608: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
      warnings.warn(
    coefficients: [ 0.21819782 95.36249001 42.64578842 -0.53980981 -0.14550926], [0.05922084]




.. GENERATED FROM PYTHON SOURCE LINES 66-67

Evaluation

.. GENERATED FROM PYTHON SOURCE LINES 67-74

.. code-block:: Python


    y_pred = clr.predict(X_test)
    sl2 = mean_squared_error(y_test, y_pred)
    sr2 = r2_score(y_test, y_pred)
    print(f"SGDRegressor: sl2={sl2}, sr2={sr2}")






.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    SGDRegressor: sl2=101.49390416781107, sr2=0.9895093817126598




.. GENERATED FROM PYTHON SOURCE LINES 75-77

Linrar Regression with pytorch
==============================

.. GENERATED FROM PYTHON SOURCE LINES 77-126

.. code-block:: Python



    class TorchLinearRegression(torch.nn.Module):
        def __init__(self, n_dims: int, n_targets: int):
            super().__init__()
            self.linear = torch.nn.Linear(n_dims, n_targets)

        def forward(self, x):
            return self.linear(x)


    def train_loop(dataloader, model, loss_fn, optimizer):
        total_loss = 0.0

        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        model.train()
        for X, y in dataloader:
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred.ravel(), y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # training loss
            total_loss += loss

        return total_loss


    model = TorchLinearRegression(X_train.shape[1], 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    device = "cpu"
    model = model.to(device)
    dataset = torch.utils.data.TensorDataset(
        torch.Tensor(X_train).to(device), torch.Tensor(y_train).to(device)
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)


    for i in range(5):
        loss = train_loop(dataloader, model, loss_fn, optimizer)
        print(f"iteration {i}, loss={loss}")





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    iteration 0, loss=2720875.5
    iteration 1, loss=196841.9375
    iteration 2, loss=85080.1953125
    iteration 3, loss=79213.7109375
    iteration 4, loss=78801.4453125




.. GENERATED FROM PYTHON SOURCE LINES 127-128

Let's check the error

.. GENERATED FROM PYTHON SOURCE LINES 128-134

.. code-block:: Python


    y_pred = model(torch.Tensor(X_test)).detach().numpy()
    tl2 = mean_squared_error(y_test, y_pred)
    tr2 = r2_score(y_test, y_pred)
    print(f"TorchLinearRegression: tl2={tl2}, tr2={tr2}")





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    TorchLinearRegression: tl2=101.36537040894598, tr2=0.9895226672258376




.. GENERATED FROM PYTHON SOURCE LINES 135-136

And the coefficients.

.. GENERATED FROM PYTHON SOURCE LINES 136-142

.. code-block:: Python


    print("coefficients:")
    for p in model.parameters():
        print(p)






.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    coefficients:
    Parameter containing:
    tensor([[ 0.1599, 95.0990, 43.1321, -0.3775,  0.2863]], requires_grad=True)
    Parameter containing:
    tensor([0.2348], requires_grad=True)




.. GENERATED FROM PYTHON SOURCE LINES 143-147

Conversion to ONNX
==================

Let's convert it to ONNX.

.. GENERATED FROM PYTHON SOURCE LINES 147-150

.. code-block:: Python


    onx = to_onnx(model, (torch.Tensor(X_test[:2]),), input_names=["x"])








.. GENERATED FROM PYTHON SOURCE LINES 151-152

Let's check it is work.

.. GENERATED FROM PYTHON SOURCE LINES 152-157

.. code-block:: Python


    sess = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
    res = sess.run(None, {"x": X_test.astype(np.float32)[:2]})
    print(res)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    [array([[-117.064224],
           [  46.55126 ]], dtype=float32)]




.. GENERATED FROM PYTHON SOURCE LINES 158-159

And the model.

.. GENERATED FROM PYTHON SOURCE LINES 159-163

.. code-block:: Python


    plot_dot(onx)





.. image-sg:: /auto_examples/images/sphx_glr_plot_torch_linreg_101_001.png
   :alt: plot torch linreg 101
   :srcset: /auto_examples/images/sphx_glr_plot_torch_linreg_101_001.png
   :class: sphx-glr-single-img





.. GENERATED FROM PYTHON SOURCE LINES 164-171

With dynamic shapes
===================

The dynamic shapes are used by :func:`torch.export.export` and must
follow the convention described there. The dynamic dimension allows
any value. The model is then valid for many different shapes.
That's usually what users need.

.. GENERATED FROM PYTHON SOURCE LINES 171-183

.. code-block:: Python



    onx = to_onnx(
        model,
        (torch.Tensor(X_test[:2]),),
        input_names=["x"],
        dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
    )

    print(pretty_onnx(onx))






.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    opset: domain='' version=18
    input: name='x' type=dtype('float32') shape=['batch', 5]
    init: name='init7_s2_-1_1' type=int64 shape=(2,) -- array([-1,  1])   -- TransposeEqualReshapePattern.apply.new_shape
    init: name='init7_s2_1_-1' type=int64 shape=(2,) -- array([ 1, -1])   -- TransposeEqualReshapePattern.apply.new_shape
    init: name='linear.weight' type=float32 shape=(1, 5)                  -- DynamoInterpret.placeholder.1/P(linear.weight)
    init: name='linear.bias' type=float32 shape=(1,) -- array([0.23478925], dtype=float32)-- DynamoInterpret.placeholder.1/P(linear.bias)
    Reshape(linear.weight, init7_s2_-1_1) -> _onx_transpose_p_linear_weight0
      Reshape(_onx_transpose_p_linear_weight0, init7_s2_1_-1) -> GemmTransposePattern--_onx_transpose_p_linear_weight0
        Gemm(x, GemmTransposePattern--_onx_transpose_p_linear_weight0, linear.bias, transB=1) -> output_0
    output: name='output_0' type=dtype('float32') shape=['batch', 1]




.. GENERATED FROM PYTHON SOURCE LINES 184-186

For simplicity, it is possible to use ``torch.export.Dim.DYNAMIC``
or ``torch.export.Dim.AUTO``.

.. GENERATED FROM PYTHON SOURCE LINES 186-195

.. code-block:: Python


    onx = to_onnx(
        model,
        (torch.Tensor(X_test[:2]),),
        input_names=["x"],
        dynamic_shapes={"x": {0: torch.export.Dim.DYNAMIC}},
    )

    print(pretty_onnx(onx))




.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    opset: domain='' version=18
    input: name='x' type=dtype('float32') shape=['batch', 5]
    init: name='init7_s2_-1_1' type=int64 shape=(2,) -- array([-1,  1])   -- TransposeEqualReshapePattern.apply.new_shape
    init: name='init7_s2_1_-1' type=int64 shape=(2,) -- array([ 1, -1])   -- TransposeEqualReshapePattern.apply.new_shape
    init: name='linear.weight' type=float32 shape=(1, 5)                  -- DynamoInterpret.placeholder.1/P(linear.weight)
    init: name='linear.bias' type=float32 shape=(1,) -- array([0.23478925], dtype=float32)-- DynamoInterpret.placeholder.1/P(linear.bias)
    Reshape(linear.weight, init7_s2_-1_1) -> _onx_transpose_p_linear_weight0
      Reshape(_onx_transpose_p_linear_weight0, init7_s2_1_-1) -> GemmTransposePattern--_onx_transpose_p_linear_weight0
        Gemm(x, GemmTransposePattern--_onx_transpose_p_linear_weight0, linear.bias, transB=1) -> output_0
    output: name='output_0' type=dtype('float32') shape=['batch', 1]





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 2.279 seconds)


.. _sphx_glr_download_auto_examples_plot_torch_linreg_101.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_torch_linreg_101.ipynb <plot_torch_linreg_101.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_torch_linreg_101.py <plot_torch_linreg_101.py>`

    .. container:: sphx-glr-download sphx-glr-download-zip

      :download:`Download zipped: plot_torch_linreg_101.zip <plot_torch_linreg_101.zip>`


.. include:: plot_torch_linreg_101.recommendations


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
