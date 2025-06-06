
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_examples/plot_onnx_diff.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_examples_plot_onnx_diff.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_plot_onnx_diff.py:


.. _l-onnx-diff-example:

Compares the conversions of the same model with different options
=================================================================

The script compares two onnx models obtained with the same trained
scikit-learn models but converted with different options.

A model
+++++++

.. GENERATED FROM PYTHON SOURCE LINES 14-28

.. code-block:: Python


    from sklearn.mixture import GaussianMixture
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from skl2onnx import to_onnx
    from onnx_array_api.reference import compare_onnx_execution
    from onnx_array_api.plotting.text_plot import onnx_simple_text_plot


    data = load_iris()
    X_train, X_test = train_test_split(data.data)
    model = GaussianMixture()
    model.fit(X_train)






.. raw:: html

    <div class="output_subarea output_html rendered_html output_result">
    <style>#sk-container-id-1 {
      /* Definition of color scheme common for light and dark mode */
      --sklearn-color-text: #000;
      --sklearn-color-text-muted: #666;
      --sklearn-color-line: gray;
      /* Definition of color scheme for unfitted estimators */
      --sklearn-color-unfitted-level-0: #fff5e6;
      --sklearn-color-unfitted-level-1: #f6e4d2;
      --sklearn-color-unfitted-level-2: #ffe0b3;
      --sklearn-color-unfitted-level-3: chocolate;
      /* Definition of color scheme for fitted estimators */
      --sklearn-color-fitted-level-0: #f0f8ff;
      --sklearn-color-fitted-level-1: #d4ebff;
      --sklearn-color-fitted-level-2: #b3dbfd;
      --sklearn-color-fitted-level-3: cornflowerblue;

      /* Specific color for light theme */
      --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
      --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
      --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
      --sklearn-color-icon: #696969;

      @media (prefers-color-scheme: dark) {
        /* Redefinition of color scheme for dark theme */
        --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
        --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
        --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
        --sklearn-color-icon: #878787;
      }
    }

    #sk-container-id-1 {
      color: var(--sklearn-color-text);
    }

    #sk-container-id-1 pre {
      padding: 0;
    }

    #sk-container-id-1 input.sk-hidden--visually {
      border: 0;
      clip: rect(1px 1px 1px 1px);
      clip: rect(1px, 1px, 1px, 1px);
      height: 1px;
      margin: -1px;
      overflow: hidden;
      padding: 0;
      position: absolute;
      width: 1px;
    }

    #sk-container-id-1 div.sk-dashed-wrapped {
      border: 1px dashed var(--sklearn-color-line);
      margin: 0 0.4em 0.5em 0.4em;
      box-sizing: border-box;
      padding-bottom: 0.4em;
      background-color: var(--sklearn-color-background);
    }

    #sk-container-id-1 div.sk-container {
      /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
         but bootstrap.min.css set `[hidden] { display: none !important; }`
         so we also need the `!important` here to be able to override the
         default hidden behavior on the sphinx rendered scikit-learn.org.
         See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
      display: inline-block !important;
      position: relative;
    }

    #sk-container-id-1 div.sk-text-repr-fallback {
      display: none;
    }

    div.sk-parallel-item,
    div.sk-serial,
    div.sk-item {
      /* draw centered vertical line to link estimators */
      background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
      background-size: 2px 100%;
      background-repeat: no-repeat;
      background-position: center center;
    }

    /* Parallel-specific style estimator block */

    #sk-container-id-1 div.sk-parallel-item::after {
      content: "";
      width: 100%;
      border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
      flex-grow: 1;
    }

    #sk-container-id-1 div.sk-parallel {
      display: flex;
      align-items: stretch;
      justify-content: center;
      background-color: var(--sklearn-color-background);
      position: relative;
    }

    #sk-container-id-1 div.sk-parallel-item {
      display: flex;
      flex-direction: column;
    }

    #sk-container-id-1 div.sk-parallel-item:first-child::after {
      align-self: flex-end;
      width: 50%;
    }

    #sk-container-id-1 div.sk-parallel-item:last-child::after {
      align-self: flex-start;
      width: 50%;
    }

    #sk-container-id-1 div.sk-parallel-item:only-child::after {
      width: 0;
    }

    /* Serial-specific style estimator block */

    #sk-container-id-1 div.sk-serial {
      display: flex;
      flex-direction: column;
      align-items: center;
      background-color: var(--sklearn-color-background);
      padding-right: 1em;
      padding-left: 1em;
    }


    /* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
    clickable and can be expanded/collapsed.
    - Pipeline and ColumnTransformer use this feature and define the default style
    - Estimators will overwrite some part of the style using the `sk-estimator` class
    */

    /* Pipeline and ColumnTransformer style (default) */

    #sk-container-id-1 div.sk-toggleable {
      /* Default theme specific background. It is overwritten whether we have a
      specific estimator or a Pipeline/ColumnTransformer */
      background-color: var(--sklearn-color-background);
    }

    /* Toggleable label */
    #sk-container-id-1 label.sk-toggleable__label {
      cursor: pointer;
      display: flex;
      width: 100%;
      margin-bottom: 0;
      padding: 0.5em;
      box-sizing: border-box;
      text-align: center;
      align-items: start;
      justify-content: space-between;
      gap: 0.5em;
    }

    #sk-container-id-1 label.sk-toggleable__label .caption {
      font-size: 0.6rem;
      font-weight: lighter;
      color: var(--sklearn-color-text-muted);
    }

    #sk-container-id-1 label.sk-toggleable__label-arrow:before {
      /* Arrow on the left of the label */
      content: "▸";
      float: left;
      margin-right: 0.25em;
      color: var(--sklearn-color-icon);
    }

    #sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
      color: var(--sklearn-color-text);
    }

    /* Toggleable content - dropdown */

    #sk-container-id-1 div.sk-toggleable__content {
      max-height: 0;
      max-width: 0;
      overflow: hidden;
      text-align: left;
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-0);
    }

    #sk-container-id-1 div.sk-toggleable__content.fitted {
      /* fitted */
      background-color: var(--sklearn-color-fitted-level-0);
    }

    #sk-container-id-1 div.sk-toggleable__content pre {
      margin: 0.2em;
      border-radius: 0.25em;
      color: var(--sklearn-color-text);
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-0);
    }

    #sk-container-id-1 div.sk-toggleable__content.fitted pre {
      /* unfitted */
      background-color: var(--sklearn-color-fitted-level-0);
    }

    #sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
      /* Expand drop-down */
      max-height: 200px;
      max-width: 100%;
      overflow: auto;
    }

    #sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
      content: "▾";
    }

    /* Pipeline/ColumnTransformer-specific style */

    #sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
      color: var(--sklearn-color-text);
      background-color: var(--sklearn-color-unfitted-level-2);
    }

    #sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
      background-color: var(--sklearn-color-fitted-level-2);
    }

    /* Estimator-specific style */

    /* Colorize estimator box */
    #sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-2);
    }

    #sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
      /* fitted */
      background-color: var(--sklearn-color-fitted-level-2);
    }

    #sk-container-id-1 div.sk-label label.sk-toggleable__label,
    #sk-container-id-1 div.sk-label label {
      /* The background is the default theme color */
      color: var(--sklearn-color-text-on-default-background);
    }

    /* On hover, darken the color of the background */
    #sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
      color: var(--sklearn-color-text);
      background-color: var(--sklearn-color-unfitted-level-2);
    }

    /* Label box, darken color on hover, fitted */
    #sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
      color: var(--sklearn-color-text);
      background-color: var(--sklearn-color-fitted-level-2);
    }

    /* Estimator label */

    #sk-container-id-1 div.sk-label label {
      font-family: monospace;
      font-weight: bold;
      display: inline-block;
      line-height: 1.2em;
    }

    #sk-container-id-1 div.sk-label-container {
      text-align: center;
    }

    /* Estimator-specific */
    #sk-container-id-1 div.sk-estimator {
      font-family: monospace;
      border: 1px dotted var(--sklearn-color-border-box);
      border-radius: 0.25em;
      box-sizing: border-box;
      margin-bottom: 0.5em;
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-0);
    }

    #sk-container-id-1 div.sk-estimator.fitted {
      /* fitted */
      background-color: var(--sklearn-color-fitted-level-0);
    }

    /* on hover */
    #sk-container-id-1 div.sk-estimator:hover {
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-2);
    }

    #sk-container-id-1 div.sk-estimator.fitted:hover {
      /* fitted */
      background-color: var(--sklearn-color-fitted-level-2);
    }

    /* Specification for estimator info (e.g. "i" and "?") */

    /* Common style for "i" and "?" */

    .sk-estimator-doc-link,
    a:link.sk-estimator-doc-link,
    a:visited.sk-estimator-doc-link {
      float: right;
      font-size: smaller;
      line-height: 1em;
      font-family: monospace;
      background-color: var(--sklearn-color-background);
      border-radius: 1em;
      height: 1em;
      width: 1em;
      text-decoration: none !important;
      margin-left: 0.5em;
      text-align: center;
      /* unfitted */
      border: var(--sklearn-color-unfitted-level-1) 1pt solid;
      color: var(--sklearn-color-unfitted-level-1);
    }

    .sk-estimator-doc-link.fitted,
    a:link.sk-estimator-doc-link.fitted,
    a:visited.sk-estimator-doc-link.fitted {
      /* fitted */
      border: var(--sklearn-color-fitted-level-1) 1pt solid;
      color: var(--sklearn-color-fitted-level-1);
    }

    /* On hover */
    div.sk-estimator:hover .sk-estimator-doc-link:hover,
    .sk-estimator-doc-link:hover,
    div.sk-label-container:hover .sk-estimator-doc-link:hover,
    .sk-estimator-doc-link:hover {
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-3);
      color: var(--sklearn-color-background);
      text-decoration: none;
    }

    div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
    .sk-estimator-doc-link.fitted:hover,
    div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
    .sk-estimator-doc-link.fitted:hover {
      /* fitted */
      background-color: var(--sklearn-color-fitted-level-3);
      color: var(--sklearn-color-background);
      text-decoration: none;
    }

    /* Span, style for the box shown on hovering the info icon */
    .sk-estimator-doc-link span {
      display: none;
      z-index: 9999;
      position: relative;
      font-weight: normal;
      right: .2ex;
      padding: .5ex;
      margin: .5ex;
      width: min-content;
      min-width: 20ex;
      max-width: 50ex;
      color: var(--sklearn-color-text);
      box-shadow: 2pt 2pt 4pt #999;
      /* unfitted */
      background: var(--sklearn-color-unfitted-level-0);
      border: .5pt solid var(--sklearn-color-unfitted-level-3);
    }

    .sk-estimator-doc-link.fitted span {
      /* fitted */
      background: var(--sklearn-color-fitted-level-0);
      border: var(--sklearn-color-fitted-level-3);
    }

    .sk-estimator-doc-link:hover span {
      display: block;
    }

    /* "?"-specific style due to the `<a>` HTML tag */

    #sk-container-id-1 a.estimator_doc_link {
      float: right;
      font-size: 1rem;
      line-height: 1em;
      font-family: monospace;
      background-color: var(--sklearn-color-background);
      border-radius: 1rem;
      height: 1rem;
      width: 1rem;
      text-decoration: none;
      /* unfitted */
      color: var(--sklearn-color-unfitted-level-1);
      border: var(--sklearn-color-unfitted-level-1) 1pt solid;
    }

    #sk-container-id-1 a.estimator_doc_link.fitted {
      /* fitted */
      border: var(--sklearn-color-fitted-level-1) 1pt solid;
      color: var(--sklearn-color-fitted-level-1);
    }

    /* On hover */
    #sk-container-id-1 a.estimator_doc_link:hover {
      /* unfitted */
      background-color: var(--sklearn-color-unfitted-level-3);
      color: var(--sklearn-color-background);
      text-decoration: none;
    }

    #sk-container-id-1 a.estimator_doc_link.fitted:hover {
      /* fitted */
      background-color: var(--sklearn-color-fitted-level-3);
    }
    </style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GaussianMixture()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>GaussianMixture</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.6/modules/generated/sklearn.mixture.GaussianMixture.html">?<span>Documentation for GaussianMixture</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted"><pre>GaussianMixture()</pre></div> </div></div></div></div>
    </div>
    <br />
    <br />

.. GENERATED FROM PYTHON SOURCE LINES 29-31

Conversion to onnx
++++++++++++++++++

.. GENERATED FROM PYTHON SOURCE LINES 31-38

.. code-block:: Python


    onx = to_onnx(
        model, X_train[:1], options={id(model): {"score_samples": True}}, target_opset=12
    )

    print(onnx_simple_text_plot(onx))





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    opset: domain='' version=12
    input: name='X' type=dtype('float64') shape=['', 4]
    init: name='Ad_Addcst' type=float64 shape=(1,) -- array([7.35150827])
    init: name='Ge_Gemmcst' type=float64 shape=(4, 4)
    init: name='Ge_Gemmcst1' type=float64 shape=(4,) -- array([-7.51792282, -7.89910213,  4.37651563,  3.02521798])
    init: name='Mu_Mulcst' type=float64 shape=(1,) -- array([-0.5])
    init: name='Ad_Addcst1' type=float64 shape=(1,) -- array([3.24465589])
    init: name='Ad_Addcst2' type=float64 shape=(1,) -- array([0.])
    Gemm(X, Ge_Gemmcst, Ge_Gemmcst1, alpha=1.00, beta=1.00) -> Ge_Y0
      ReduceSumSquare(Ge_Y0, axes=[1], keepdims=1) -> Re_reduced0
        Concat(Re_reduced0, axis=1) -> Co_concat_result0
          Add(Ad_Addcst, Co_concat_result0) -> Ad_C02
            Mul(Ad_C02, Mu_Mulcst) -> Mu_C0
              Add(Mu_C0, Ad_Addcst1) -> Ad_C01
                Add(Ad_C01, Ad_Addcst2) -> Ad_C0
                  ArgMax(Ad_C0, axis=1) -> label
                  ReduceLogSumExp(Ad_C0, axes=[1], keepdims=1) -> score_samples
                  Sub(Ad_C0, score_samples) -> Su_C0
                    Exp(Su_C0) -> probabilities
    output: name='label' type=dtype('int64') shape=['', 1]
    output: name='probabilities' type=dtype('float64') shape=['', 1]
    output: name='score_samples' type=dtype('float64') shape=['', 1]




.. GENERATED FROM PYTHON SOURCE LINES 39-41

Conversion to onnx without ReduceLogSumExp
++++++++++++++++++++++++++++++++++++++++++

.. GENERATED FROM PYTHON SOURCE LINES 41-53

.. code-block:: Python


    onx2 = to_onnx(
        model,
        X_train[:1],
        options={id(model): {"score_samples": True}},
        black_op={"ReduceLogSumExp"},
        target_opset=12,
    )

    print(onnx_simple_text_plot(onx2))






.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    opset: domain='' version=12
    input: name='X' type=dtype('float64') shape=['', 4]
    init: name='Ad_Addcst' type=float64 shape=(1,) -- array([7.35150827])
    init: name='Ge_Gemmcst' type=float64 shape=(4, 4)
    init: name='Ge_Gemmcst1' type=float64 shape=(4,) -- array([-7.51792282, -7.89910213,  4.37651563,  3.02521798])
    init: name='Mu_Mulcst' type=float64 shape=(1,) -- array([-0.5])
    init: name='Ad_Addcst1' type=float64 shape=(1,) -- array([3.24465589])
    init: name='Ad_Addcst2' type=float64 shape=(1,) -- array([0.])
    Gemm(X, Ge_Gemmcst, Ge_Gemmcst1, alpha=1.00, beta=1.00) -> Ge_Y0
      Mul(Ge_Y0, Ge_Y0) -> Mu_C01
        ReduceSum(Mu_C01, axes=[1], keepdims=1) -> Re_reduced0
          Concat(Re_reduced0, axis=1) -> Co_concat_result0
            Add(Ad_Addcst, Co_concat_result0) -> Ad_C02
              Mul(Ad_C02, Mu_Mulcst) -> Mu_C0
                Add(Mu_C0, Ad_Addcst1) -> Ad_C01
                  Add(Ad_C01, Ad_Addcst2) -> Ad_C0
                    ArgMax(Ad_C0, axis=1) -> label
                    ReduceMax(Ad_C0, axes=[1], keepdims=1) -> Re_reduced03
                    Sub(Ad_C0, Re_reduced03) -> Su_C01
                      Exp(Su_C01) -> Ex_output0
                        ReduceSum(Ex_output0, axes=[1], keepdims=1) -> Re_reduced02
                          Log(Re_reduced02) -> Lo_output0
                      Add(Lo_output0, Re_reduced03) -> score_samples
                    Sub(Ad_C0, score_samples) -> Su_C0
                      Exp(Su_C0) -> probabilities
    output: name='label' type=dtype('int64') shape=['', 1]
    output: name='probabilities' type=dtype('float64') shape=['', 1]
    output: name='score_samples' type=dtype('float64') shape=['', 1]




.. GENERATED FROM PYTHON SOURCE LINES 54-60

Differences
+++++++++++

Function :func:`onnx_array_api.reference.compare_onnx_execution`
compares the intermediate results of two onnx models. Then it finds
the best alignmet between the two models using an edit distance.

.. GENERATED FROM PYTHON SOURCE LINES 60-66

.. code-block:: Python


    res1, res2, align, dc = compare_onnx_execution(onx, onx2, verbose=1)
    print("------------")
    text = dc.to_str(res1, res2, align)
    print(text)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    [compare_onnx_execution] generate inputs
    [compare_onnx_execution] execute with 1 inputs
    [compare_onnx_execution] execute first model
    [compare_onnx_execution] got 21 results
    [compare_onnx_execution] execute second model
    [compare_onnx_execution] got 21 results (first model)
    [compare_onnx_execution] got 27 results (second model)
    [compare_onnx_execution] compute edit distance
    [compare_onnx_execution] got 27 pairs
    [compare_onnx_execution] done
    ------------
    001 = | INITIA float64  1:1                  HAAA                 Ad | INITIA float64  1:1                  HAAA                 Ad
    002 = | INITIA float64  2:4x4                ADZF                 Ge | INITIA float64  2:4x4                ADZF                 Ge
    003 = | INITIA float64  1:4                  TTED                 Ge | INITIA float64  1:4                  TTED                 Ge
    004 = | INITIA float64  1:1                  AAAA                 Mu | INITIA float64  1:1                  AAAA                 Mu
    005 = | INITIA float64  1:1                  DAAA                 Ad | INITIA float64  1:1                  DAAA                 Ad
    006 = | INITIA float64  1:1                  AAAA                 Ad | INITIA float64  1:1                  AAAA                 Ad
    007 = | INPUT  float64  2:1x4                AAAA                 X  | INPUT  float64  2:1x4                AAAA                 X 
    008 = | RESULT float64  2:1x4                TTFF Gemm            Ge | RESULT float64  2:1x4                TTFF Gemm            Ge
    009 + |                                                              | RESULT float64  2:1x4                EBFD Mul             Mu 
    010 ~ | RESULT float64  2:1x1                PAAA ReduceSumSquare Re | RESULT float64  2:1x1                PAAA ReduceSum       Re
    011 = | RESULT float64  2:1x1                PAAA Concat          Co | RESULT float64  2:1x1                PAAA Concat          Co
    012 = | RESULT float64  2:1x1                XAAA Add             Ad | RESULT float64  2:1x1                XAAA Add             Ad
    013 = | RESULT float64  2:1x1                PAAA Mul             Mu | RESULT float64  2:1x1                PAAA Mul             Mu
    014 = | RESULT float64  2:1x1                SAAA Add             Ad | RESULT float64  2:1x1                SAAA Add             Ad
    015 = | RESULT float64  2:1x1                SAAA Add             Ad | RESULT float64  2:1x1                SAAA Add             Ad
    016 = | RESULT int64    2:1x1                AAAA ArgMax          la | RESULT int64    2:1x1                AAAA ArgMax          la
    017 + |                                                              | RESULT float64  2:1x1                SAAA ReduceMax       Re 
    018 + |                                                              | RESULT float64  2:1x1                AAAA Sub             Su 
    019 + |                                                              | RESULT float64  2:1x1                BAAA Exp             Ex 
    020 + |                                                              | RESULT float64  2:1x1                BAAA ReduceSum       Re 
    021 + |                                                              | RESULT float64  2:1x1                AAAA Log             Lo 
    022 ~ | RESULT float64  2:1x1                SAAA ReduceLogSumExp sc | RESULT float64  2:1x1                SAAA Add             sc
    023 = | RESULT float64  2:1x1                AAAA Sub             Su | RESULT float64  2:1x1                AAAA Sub             Su
    024 = | RESULT float64  2:1x1                BAAA Exp             pr | RESULT float64  2:1x1                BAAA Exp             pr
    025 = | OUTPUT int64    2:1x1                AAAA                 la | OUTPUT int64    2:1x1                AAAA                 la
    026 = | OUTPUT float64  2:1x1                BAAA                 pr | OUTPUT float64  2:1x1                BAAA                 pr
    027 = | OUTPUT float64  2:1x1                SAAA                 sc | OUTPUT float64  2:1x1                SAAA                 sc




.. GENERATED FROM PYTHON SOURCE LINES 67-70

See :ref:`l-long-output-compare_onnx_execution` for a better view.
The display shows that ReduceSumSquare was replaced by Mul + ReduceSum,
and ReduceLogSumExp by ReduceMax + Sub + Exp + Log + Add.


.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 4.456 seconds)


.. _sphx_glr_download_auto_examples_plot_onnx_diff.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_onnx_diff.ipynb <plot_onnx_diff.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_onnx_diff.py <plot_onnx_diff.py>`

    .. container:: sphx-glr-download sphx-glr-download-zip

      :download:`Download zipped: plot_onnx_diff.zip <plot_onnx_diff.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
