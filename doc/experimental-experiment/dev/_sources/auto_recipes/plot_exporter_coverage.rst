
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_recipes/plot_exporter_coverage.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_recipes_plot_exporter_coverage.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_recipes_plot_exporter_coverage.py:


.. _l-plot-exporter-coverage:

Measures the exporter success on many test cases
================================================

All test cases can be found in module
:mod:`experimental_experiment.torch_interpreter.eval.model_cases`.
Page :ref:`l-export-supported-signatures` shows the exported
program for many of those cases.

.. GENERATED FROM PYTHON SOURCE LINES 13-63

.. code-block:: Python


    from experimental_experiment.args import get_parsed_args

    script_args = get_parsed_args(
        "plot_exporter_coverage",
        description=__doc__,
        exporter=("all", "an exporter to rerun"),
        dynamic=("all", "use dyanmic shapes"),
        case=(
            "three",
            "model cases, two for the first two (to test), "
            "all to select all, a name or a regular expression fior a subset",
        ),
        quiet=("1", "0 or 1"),
        verbose=("1", "verbosity"),
        expose="exporter,dyanmic,case,quiet,verbose",
    )

    exporters = (
        (
            "export-strict",
            "export-strict-dec",
            "export-nostrict",
            "export-nostrict-dec",
            "export-jit",
            "export-tracing",
            "custom-strict",
            "custom-nostrict",
            "custom-strict-dec",
            "custom-nostrict-dec",
            "custom-tracing",
            "dynamo",
            "dynamo-ir",
            "script",
        )
        if script_args.exporter == "all"
        else script_args.exporter.split(",")
    )
    dynamic = (0, 1) if script_args.dynamic == "all" else (int(script_args.dynamic),)
    cases = None if script_args.case == "all" else script_args.case.split(",")
    quiet = bool(int(script_args.quiet))
    verbose = int(script_args.verbose)

    import pandas
    from experimental_experiment.torch_interpreter.eval import evaluation

    obs = evaluation(
        exporters=exporters, dynamic=dynamic, cases=cases, quiet=quiet, verbose=verbose
    )





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

      0%|          | 0/84 [00:00<?, ?it/s]      2%|▏         | 2/84 [00:00<00:09,  8.82it/s]      5%|▍         | 4/84 [00:00<00:07, 10.34it/s]      7%|▋         | 6/84 [00:00<00:06, 12.47it/s]     11%|█         | 9/84 [00:00<00:05, 12.63it/s]     13%|█▎        | 11/84 [00:01<00:07, 10.07it/s]     15%|█▌        | 13/84 [00:02<00:24,  2.90it/s]     18%|█▊        | 15/84 [00:02<00:17,  3.84it/s]     19%|█▉        | 16/84 [00:03<00:16,  4.10it/s]     21%|██▏       | 18/84 [00:03<00:12,  5.18it/s]     26%|██▌       | 22/84 [00:03<00:07,  8.11it/s]     29%|██▊       | 24/84 [00:04<00:11,  5.14it/s]     31%|███       | 26/84 [00:04<00:13,  4.36it/s]     32%|███▏      | 27/84 [00:05<00:18,  3.11it/s]     35%|███▍      | 29/84 [00:05<00:13,  4.06it/s]     36%|███▌      | 30/84 [00:06<00:16,  3.35it/s]     38%|███▊      | 32/84 [00:06<00:12,  4.20it/s]     40%|████      | 34/84 [00:06<00:08,  5.66it/s]     44%|████▍     | 37/84 [00:07<00:06,  7.11it/s]     46%|████▋     | 39/84 [00:07<00:05,  7.73it/s]     49%|████▉     | 41/84 [00:09<00:17,  2.42it/s]     51%|█████     | 43/84 [00:09<00:13,  3.13it/s]     52%|█████▏    | 44/84 [00:09<00:12,  3.14it/s]     55%|█████▍    | 46/84 [00:10<00:09,  3.86it/s]     58%|█████▊    | 49/84 [00:10<00:06,  5.43it/s]     61%|██████    | 51/84 [00:10<00:05,  5.66it/s]     62%|██████▏   | 52/84 [00:11<00:08,  3.70it/s]     64%|██████▍   | 54/84 [00:12<00:09,  3.08it/s]     65%|██████▌   | 55/84 [00:13<00:13,  2.12it/s]     68%|██████▊   | 57/84 [00:13<00:08,  3.01it/s]     69%|██████▉   | 58/84 [00:13<00:07,  3.30it/s]     71%|███████▏  | 60/84 [00:14<00:06,  3.89it/s]     76%|███████▌  | 64/84 [00:14<00:03,  6.65it/s]     79%|███████▊  | 66/84 [00:15<00:04,  4.43it/s]     81%|████████  | 68/84 [00:16<00:04,  3.61it/s]     82%|████████▏ | 69/84 [00:16<00:05,  2.85it/s]     86%|████████▌ | 72/84 [00:17<00:03,  3.39it/s]     88%|████████▊ | 74/84 [00:17<00:02,  4.14it/s]     94%|█████████▍| 79/84 [00:17<00:00,  6.92it/s]     96%|█████████▋| 81/84 [00:18<00:00,  7.05it/s]     98%|█████████▊| 82/84 [00:18<00:00,  5.03it/s]     99%|█████████▉| 83/84 [00:19<00:00,  3.29it/s]    100%|██████████| 84/84 [00:19<00:00,  4.29it/s]




.. GENERATED FROM PYTHON SOURCE LINES 64-65

The results

.. GENERATED FROM PYTHON SOURCE LINES 65-74

.. code-block:: Python


    df = pandas.DataFrame(obs).sort_values(["dynamic", "name", "exporter"]).reset_index(drop=True)
    df.to_csv("plot-exporter-coverage.csv", index=False)
    df.to_excel("plot-exporter-coverage.xlsx")
    for c in ["error", "error_step"]:
        if c in df.columns:
            df[c] = df[c].fillna("")
    print(df)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

        abs  rel  dnan       argm  success                                          model_cls  ...                                               onnx         name dynamic             exporter                                              error error_step
    0   0.0  0.0   0.0  (0, 0, 0)        1  <class 'experimental_experiment.torch_interpre...  ...  ir_version: 8\ngraph {\n  node {\n    input: "...  AtenRollPos       0      custom-nostrict                                                              
    1   0.0  0.0   0.0  (0, 0, 0)        1  <class 'experimental_experiment.torch_interpre...  ...  ir_version: 8\ngraph {\n  node {\n    input: "...  AtenRollPos       0  custom-nostrict-dec                                                              
    2   0.0  0.0   0.0  (0, 0, 0)        1  <class 'experimental_experiment.torch_interpre...  ...  ir_version: 8\ngraph {\n  node {\n    input: "...  AtenRollPos       0        custom-strict                                                              
    3   0.0  0.0   0.0  (0, 0, 0)        1  <class 'experimental_experiment.torch_interpre...  ...  ir_version: 8\ngraph {\n  node {\n    input: "...  AtenRollPos       0    custom-strict-dec                                                              
    4   0.0  0.0   0.0  (0, 0, 0)        1  <class 'experimental_experiment.torch_interpre...  ...  ir_version: 8\ngraph {\n  node {\n    input: "...  AtenRollPos       0       custom-tracing                                                              
    ..  ...  ...   ...        ...      ...                                                ...  ...                                                ...          ...     ...                  ...                                                ...        ...
    79  0.0  0.0   0.0     (0, 0)        1  <class 'experimental_experiment.torch_interpre...  ...                                               None   InplaceAdd       1  export-nostrict-dec                                                              
    80  0.0  0.0   0.0     (0, 0)        1  <class 'experimental_experiment.torch_interpre...  ...                                               None   InplaceAdd       1        export-strict                                                              
    81  0.0  0.0   0.0     (0, 0)        1  <class 'experimental_experiment.torch_interpre...  ...                                               None   InplaceAdd       1    export-strict-dec                                                              
    82  0.0  0.0   0.0     (0, 0)        1  <class 'experimental_experiment.torch_interpre...  ...                                               None   InplaceAdd       1       export-tracing                                                              
    83  NaN  NaN   NaN        NaN        0                                                NaN  ...                                                NaN   InplaceAdd       1               script  number of input names provided (3) exceeded nu...     export

    [84 rows x 13 columns]




.. GENERATED FROM PYTHON SOURCE LINES 75-76

Errors if any or all successes.

.. GENERATED FROM PYTHON SOURCE LINES 76-85

.. code-block:: Python


    piv = df.pivot(
        index=["dynamic", "name"],
        columns=["exporter"],
        values="error_step" if "error_step" in df.columns else "success",
    )

    piv.to_excel("plot-exporter-coverage-summary.xlsx")
    print(piv)




.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    exporter             custom-nostrict custom-nostrict-dec custom-strict custom-strict-dec custom-tracing dynamo dynamo-ir export-jit export-nostrict export-nostrict-dec export-strict export-strict-dec export-tracing  script
    dynamic name                                                                                                                                                                                                                  
    0       AtenRollPos                                                                                                                                                                                                           
            AtenRollRelu                                                                                                                                                                                                          
            InplaceAdd                                                                                                                                                                                                            
    1       AtenRollPos                                                                                                                                                                                                     export
            AtenRollRelu                                                                                                                                                                                                    export
            InplaceAdd                                                                                                                                                                                                      export





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 19.781 seconds)


.. _sphx_glr_download_auto_recipes_plot_exporter_coverage.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_exporter_coverage.ipynb <plot_exporter_coverage.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_exporter_coverage.py <plot_exporter_coverage.py>`

    .. container:: sphx-glr-download sphx-glr-download-zip

      :download:`Download zipped: plot_exporter_coverage.zip <plot_exporter_coverage.zip>`


.. include:: plot_exporter_coverage.recommendations


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
