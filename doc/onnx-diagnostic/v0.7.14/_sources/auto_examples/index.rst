:orphan:

Examples Gallery
================



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Looking for discrepancies is quickly annoying. Discrepancies come from two results obtained with the same models implemented in two different ways, pytorch and onnx. Models are big so where do they come from? That&#x27;s the unavoidable question. Unless there is an obvious reason, the only way is to compare intermediate outputs alon the computation. The first step into that direction is to dump the intermediate results coming from pytorch. We use onnx_diagnostic.helpers.torch_helper.steal_forward for that.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_dump_intermediate_results_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_dump_intermediate_results.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dumps intermediate results of a torch model</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Quick tour of dynamic shapes. We first look at examples playing positional and names parameters to understand how torch.export.export works.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_export_with_args_kwargs_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_export_with_args_kwargs.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dynamic Shapes for args, *kwargs</div>
    </div>


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

    <div class="sphx-glr-thumbcontainer" tooltip="This function exports an smaller untrained model with the same architecture. It is faster than the pretrained model. When this works, the untrained model can be replaced by the trained one.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_export_tiny_phi2_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_export_tiny_phi2.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Export microsoft/phi-2</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Every LLMs implemented in transformers use cache. One of the most used is transformers.cache_utils.DynamicCache. The cache size is dynamic to cope with the growing context. The example shows a tool which determines the dynamic shapes for torch.export.export based on a set of valid inputs.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_export_with_dynamic_cache_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_export_with_dynamic_cache.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Export with DynamicCache and guessed dynamic shapes</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The first version of torch.export.export did not support any tensor with a dimension equal to 0, 1 if the dimension was expected to be dynamic. The latest versions offers more options. Let&#x27;s check it works. The experiments consists in exporting the model with different sets of inputs and checking the exported models works with all set of inputs.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_export_tiny_llm_dim01_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_export_tiny_llm_dim01.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Export with dynamic dimensions in {0,1}</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This duplicates the example l-plot-tiny-llm-export-dim01 but for torch.onnx.export. It checks what inputs can be used to export and with which inputs it can work.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_export_tiny_llm_dim01_onnx_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_export_tiny_llm_dim01_onnx.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Export with dynamic dimensions in {0,1} into ONNX</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This duplicates the example l-plot-tiny-llm-export-dim01 but for experimental_experiment.torch_interpreter.to_onnx. It checks what inputs can be used to export and with which inputs it can work.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_export_tiny_llm_dim01_onnx_custom_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_export_tiny_llm_dim01_onnx_custom.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Export with dynamic dimensions in {0,1} into ONNX (custom)</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="Example l-plot-failing-reference-evaluator demonstrated how to run a python runtime on a model but it may very slow sometimes and it could show some discrepancies if the only provider is not CPU. Let&#x27;s use onnx_diagnostic.reference.OnnxruntimeEvaluator. It splits the model into node and runs them independently until it succeeds or fails. This class converts every node into model based on the types discovered during the execution. It relies on onnx_diagnostic.helpers.ort_session.InferenceSessionForTorch or onnx_diagnostic.helpers.ort_session.InferenceSessionForNumpy for the execution. This example uses torch tensor and bfloat16.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_failing_onnxruntime_evaluator_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_failing_onnxruntime_evaluator.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Intermediate results with onnxruntime</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Inputs are always dynamic with LLMs that is why dynamic shapes needs to be specified when a LLM is exported with torch.export.export. Most of the examples on HuggingFace use method transformers.GenerationMixin.generate but we only want to export the model and its method forward.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_export_tiny_llm_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_export_tiny_llm.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Steel method forward to guess inputs and dynamic shapes (with Tiny-LLM)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Checking the exporter on a whole model takes time as it is usually big but we can create a smaller version with the same architecture. Then fix export issues on such a small model is faster.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_export_hub_codellama_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_plot_export_hub_codellama.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Test the export on untrained models</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/plot_dump_intermediate_results
   /auto_examples/plot_export_with_args_kwargs
   /auto_examples/plot_export_tiny_llm_patched
   /auto_examples/plot_export_tiny_phi2
   /auto_examples/plot_export_with_dynamic_cache
   /auto_examples/plot_export_tiny_llm_dim01
   /auto_examples/plot_export_tiny_llm_dim01_onnx
   /auto_examples/plot_export_tiny_llm_dim01_onnx_custom
   /auto_examples/plot_export_locate_issue
   /auto_examples/plot_failing_model_extract
   /auto_examples/plot_failing_reference_evaluator
   /auto_examples/plot_failing_onnxruntime_evaluator
   /auto_examples/plot_export_tiny_llm
   /auto_examples/plot_export_hub_codellama


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: auto_examples_python.zip </auto_examples/auto_examples_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: auto_examples_jupyter.zip </auto_examples/auto_examples_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
