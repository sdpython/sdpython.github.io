:orphan:

Exporter Recipes Gallery
========================

A model can be converted to ONNX if :func:`torch.export.export` is
able to convert that model into a graph. This is not always
possible but usually possible with some code changes.
These changes may not be desired as they may hurt the performance
and make the code more complex than it should be.
The conversion is a necessary step to be able to use
ONNX. Next examples shows some recurrent code patterns and
ways to rewrite them so that the exporter works.

See :ref:`l-exporter-recipes` for an organized version of this gallery.

A couple of examples to illustrate different implementation
of dot product (see also :epkg:`sphinx-gallery`).

Getting started
+++++++++++++++

pytorch nightly build should be installed, see
`Start Locally <https://pytorch.org/get-started/locally/>`_.

::

    git clone https://github.com/sdpython/experimental-experiment.git
    pip install onnxruntime-gpu pynvml
    pip install -r requirements-dev.txt    
    export PYTHONPATH=$PYTHONPATH:<this folder>

Common Errors
+++++++++++++

Some of them are exposed in the examples. Others may be found at
:ref:`l-frequent-exporter-errors`.




.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Dynamic shapes ensures a model is valid not matter what the dimension value is for a dynamic dimension. torch.export.export is trying to keep track of that information for every intermediate result the model produces. But something it fails. Let&#x27;s see one case.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_exporter_exporter_lost_dynamic_dimension_thumb.png
    :alt:

  :ref:`sphx_glr_auto_recipes_plot_exporter_exporter_lost_dynamic_dimension.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A dynamic dimension lost by torch.export.export</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This continues example l-plot-torch-export-with-dynamic-cache-201.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_exporter_exporter_inputs_thumb.png
    :alt:

  :ref:`sphx_glr_auto_recipes_plot_exporter_exporter_inputs.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Do no use Module as inputs!</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="torch.export.export often breaks on big models because there are control flows or instructions breaking the propagation of dynamic shapes (see ...). The function usually gives an indication where the model implementation can be fixed but in case, that is not possible, we can try to export the model piece by piece: every module is converted separately from its submodule. A model can be exported even if one of its submodules cannot.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_exporter_exporter_phi35_piece_thumb.png
    :alt:

  :ref:`sphx_glr_auto_recipes_plot_exporter_exporter_phi35_piece.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Export Phi-3.5-mini-instruct piece by piece</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Tries torch.export._draft_export.draft_export.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_exporter_exporter_draft_mode_thumb.png
    :alt:

  :ref:`sphx_glr_auto_recipes_plot_exporter_exporter_draft_mode.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Export Phi-3.5-mini-instruct with draft_export</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Tries torch._export.tools.report_exportability.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_exporter_exporter_reportibility_thumb.png
    :alt:

  :ref:`sphx_glr_auto_recipes_plot_exporter_exporter_reportibility.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Export Phi-3.5-mini-instruct with report_exportability</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="We will a class used in many model: transformers.cache_utils.DynamicCache.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_exporter_exporter_with_dynamic_cache_thumb.png
    :alt:

  :ref:`sphx_glr_auto_recipes_plot_exporter_exporter_with_dynamic_cache.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Export a model using a custom type as input</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Control flow cannot be exported with a change. The code of the model can be changed or patched to introduce function torch.ops.higher_order.scan.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_exporter_exporter_scan_pdist_thumb.png
    :alt:

  :ref:`sphx_glr_auto_recipes_plot_exporter_exporter_scan_pdist.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Export a model with a loop (scan)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Dynamic shapes need to be specified to get a model able to cope with different dimensions. Input rank are expected to be the same but the dimension may change. The user has the ability to set them up or to call a function able to infer them from two sets of inputs having different values for the dynamic dimensions.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_exporter_exporter_infer_ds_thumb.png
    :alt:

  :ref:`sphx_glr_auto_recipes_plot_exporter_exporter_infer_ds.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Infer dynamic shapes before exporting</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="scikit-learn and torch to train a linear regression.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_exporter_recipes_oe_lr_thumb.png
    :alt:

  :ref:`sphx_glr_auto_recipes_plot_exporter_recipes_oe_lr.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Linear Regression and export to ONNX</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="All test cases can be found in module experimental_experiment.torch_interpreter.eval.model_cases. Page l-export-supported-signatures shows the exported program for many of those cases.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_exporter_coverage_thumb.png
    :alt:

  :ref:`sphx_glr_auto_recipes_plot_exporter_coverage.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Measures the exporter success on many test cases</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Settings the dynamic shapes is not always easy. Here are a few tricks to make it work.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_exporter_exporter_dynamic_shapes_auto_thumb.png
    :alt:

  :ref:`sphx_glr_auto_recipes_plot_exporter_exporter_dynamic_shapes_auto.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Use DYNAMIC or AUTO when dynamic shapes has constraints</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Exports model Phi-2. We use a dummy model. The main difficulty is to set the dynamic shapes properly. If there is an issue, you can go to the following line: torch/fx/experimental/symbolic_shapes.py#L5965 and look for log.info(&quot;set_replacement %s = %s (%s) %s&quot;, a, tgt, msg, tgt_bound) and add before or after, something like:">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_exporter_recipes_c_phi2_thumb.png
    :alt:

  :ref:`sphx_glr_auto_recipes_plot_exporter_recipes_c_phi2.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">to_onnx and Phi-2</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example shows how to convert a custom operator as defined in the tutorial Python Custom Operators.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_exporter_recipes_c_custom_ops_inplace_thumb.png
    :alt:

  :ref:`sphx_glr_auto_recipes_plot_exporter_recipes_c_custom_ops_inplace.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">to_onnx and a custom operator inplace</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example shows how to convert a custom operator, inspired from Python Custom Operators.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_exporter_recipes_c_custom_ops_fct_thumb.png
    :alt:

  :ref:`sphx_glr_auto_recipes_plot_exporter_recipes_c_custom_ops_fct.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">to_onnx and a custom operator registered with a function</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Control flow cannot be exported with a change. The code of the model can be changed or patched to introduce function torch.ops.higher_order.scan.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_exporter_recipes_c_scan_pdist_thumb.png
    :alt:

  :ref:`sphx_glr_auto_recipes_plot_exporter_recipes_c_scan_pdist.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">to_onnx and a model with a loop (scan)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Control flow cannot be exported with a change. The code of the model can be changed or patched to introduce function torch.cond.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_exporter_recipes_c_cond_thumb.png
    :alt:

  :ref:`sphx_glr_auto_recipes_plot_exporter_recipes_c_cond.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">to_onnx and a model with a test</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This is a frequent task which does not play well with dynamic shapes. Let&#x27;s see how to avoid using torch.cond.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_exporter_recipes_c_dynpad_thumb.png
    :alt:

  :ref:`sphx_glr_auto_recipes_plot_exporter_recipes_c_dynpad.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">to_onnx and padding one dimension to a mulitple of a constant</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Big models are hard to read once converted into onnx. Let&#x27;s see how to improve their readibility. The code is inspired from LLM from scratch with Pytorch.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_exporter_recipes_c_modules_thumb.png
    :alt:

  :ref:`sphx_glr_auto_recipes_plot_exporter_recipes_c_modules.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">to_onnx and submodules from LLMs</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Example given in l-plot-exporter-dynamic_shapes can only be exported with dynamic shapes using torch.export.Dim.AUTO. As a result, the exported onnx models have dynamic dimensions with unpredictable names.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_exporter_recipes_c_named_ds_auto_thumb.png
    :alt:

  :ref:`sphx_glr_auto_recipes_plot_exporter_recipes_c_named_ds_auto.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">to_onnx: Rename Dynamic Shapes</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Exports model Phi-2. We use a dummy model. The main difficulty is to set the dynamic shapes properly.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_exporter_recipes_oe_phi2_thumb.png
    :alt:

  :ref:`sphx_glr_auto_recipes_plot_exporter_recipes_oe_phi2.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">torch.onnx.export and Phi-2</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Tests cannot be exported into ONNX unless they refactored to use torch.cond.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_exporter_recipes_oe_cond_thumb.png
    :alt:

  :ref:`sphx_glr_auto_recipes_plot_exporter_recipes_oe_cond.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">torch.onnx.export and a model with a test</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This is a frequent task which does not play well with dynamic shapes. Let&#x27;s see how to avoid using torch.cond.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_exporter_recipes_oe_dynpad_thumb.png
    :alt:

  :ref:`sphx_glr_auto_recipes_plot_exporter_recipes_oe_dynpad.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">torch.onnx.export and padding one dimension to a mulitple of a constant</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Example given in l-plot-exporter-dynamic_shapes can only be exported with dynamic shapes using torch.export.Dim.AUTO. As a result, the exported onnx models have dynamic dimensions with unpredictable names.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_exporter_recipes_oe_named_ds_auto_thumb.png
    :alt:

  :ref:`sphx_glr_auto_recipes_plot_exporter_recipes_oe_named_ds_auto.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">torch.onnx.export: Rename Dynamic Shapes</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_recipes/plot_exporter_exporter_lost_dynamic_dimension
   /auto_recipes/plot_exporter_exporter_inputs
   /auto_recipes/plot_exporter_exporter_phi35_piece
   /auto_recipes/plot_exporter_exporter_draft_mode
   /auto_recipes/plot_exporter_exporter_reportibility
   /auto_recipes/plot_exporter_exporter_with_dynamic_cache
   /auto_recipes/plot_exporter_exporter_scan_pdist
   /auto_recipes/plot_exporter_exporter_infer_ds
   /auto_recipes/plot_exporter_recipes_oe_lr
   /auto_recipes/plot_exporter_coverage
   /auto_recipes/plot_exporter_exporter_dynamic_shapes_auto
   /auto_recipes/plot_exporter_recipes_c_phi2
   /auto_recipes/plot_exporter_recipes_c_custom_ops_inplace
   /auto_recipes/plot_exporter_recipes_c_custom_ops_fct
   /auto_recipes/plot_exporter_recipes_c_scan_pdist
   /auto_recipes/plot_exporter_recipes_c_cond
   /auto_recipes/plot_exporter_recipes_c_dynpad
   /auto_recipes/plot_exporter_recipes_c_modules
   /auto_recipes/plot_exporter_recipes_c_named_ds_auto
   /auto_recipes/plot_exporter_recipes_oe_phi2
   /auto_recipes/plot_exporter_recipes_oe_cond
   /auto_recipes/plot_exporter_recipes_oe_dynpad
   /auto_recipes/plot_exporter_recipes_oe_named_ds_auto


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: auto_recipes_python.zip </auto_recipes/auto_recipes_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: auto_recipes_jupyter.zip </auto_recipes/auto_recipes_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
