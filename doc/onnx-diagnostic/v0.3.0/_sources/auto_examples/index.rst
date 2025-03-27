:orphan:

Examples Gallery
================



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Many models from transformers cannot be converted because the implementation uses cache classes. Let&#x27;s see how to get around that. We focus on the model arnir0/Tiny-LLM. To avoid downloading any weights, we write a function creating a random model based on the same architecture. This continues example l-plot-tiny-llm-export.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_export_tiny_llm_patched_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_export_tiny_llm_patched.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Export Tiny-LLM with patches</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Control flow cannot be exported with a change. The code of the model can be changed or patched to introduce function torch.cond.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_export_cond_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_export_cond.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Export a model with a control flow (If)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Every LLMs implemented in transformers use cache. One of the most used is transformers.cache_utils.DynamicCache. The cache size is dynamic to cope with the growing context. The example shows a tool which determines the dynamic shapes for torch.export.export based on a set of valid inputs.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_export_with_dynamic_cache_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_export_with_dynamic_cache.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Export with DynamicCache and dynamic shapes</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="LLMs must be exported with dynamic shapes and it is common that a static dimension turns into a static ones. The error message from pytorch tells the user to define TORCH_LOGS=&quot;+dynamic&quot; but it shows a very long list of messages where we need to find the string range_refined_to_singleton and that does not really indicates where it comes from. The example shows how to tweak pytorch to get that information until it gets better.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_export_locate_issue_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_export_locate_issue.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Find and fix an export issue due to dynamic shapes</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Let&#x27;s assume onnxruntime crashes without telling why or where. The first thing is do is to locate where. For that, we extract every submodel starting from the inputs and running the first n nodes of the model. The model is likely to fail for some n. Then the failing is known.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_failing_model_extract_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_failing_model_extract.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Find where a model is failing by running submodels</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Let&#x27;s assume onnxruntime crashes without telling why or where. The first thing is do is to locate where. For that, we run a python runtime which is going to run until it fails.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_failing_reference_evaluator_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_failing_reference_evaluator.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Intermediate results with (ONNX) ReferenceEvaluator</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Example l-plot-failing-reference-evaluator demonstrated how to run a python runtime on a model but it may very slow sometimes and it could show some discrepancies if the only provider is not CPU. Let&#x27;s use onnx_diagnostic.reference.OnnxruntimeEvaluator. It splits the model into node and runs them independently until it succeeds or fails. This class converts every node into model based on the types discovered during the execution. It relies on onnx_diagnostic.ort_session.InferenceSessionForTorch or onnx_diagnostic.ort_session.InferenceSessionForNumpy for the execution. This example uses torch tensor and bfloat16.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_failing_onnxruntime_evaluator_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_failing_onnxruntime_evaluator.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Intermediate results with onnxruntime</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Inputs are always dynamic with LLMs that is why dynamic shapes needs to be specified when a LLM is exported withtorch.export.export. Most of the examples on HuggingFace use method transformers.GenerationMixin.generate but we only want to export the model and its method forward.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_export_tiny_llm_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_export_tiny_llm.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Steel method forward to guess the dynamic shapes (with Tiny-LLM)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Setting the dynamic shapes is not always easy. Here are a few tricks to make it work.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_export_with_dynamic_shapes_auto_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_export_with_dynamic_shapes_auto.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Use DYNAMIC or AUTO when exporting if dynamic shapes has constraints</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/plot_export_tiny_llm_patched
   /auto_examples/plot_export_cond
   /auto_examples/plot_export_with_dynamic_cache
   /auto_examples/plot_export_locate_issue
   /auto_examples/plot_failing_model_extract
   /auto_examples/plot_failing_reference_evaluator
   /auto_examples/plot_failing_onnxruntime_evaluator
   /auto_examples/plot_export_tiny_llm
   /auto_examples/plot_export_with_dynamic_shapes_auto


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: auto_examples_python.zip </auto_examples/auto_examples_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: auto_examples_jupyter.zip </auto_examples/auto_examples_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
