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

A couple of examples to illustrate different implementation
of dot product (see also :epkg:`sphinx-gallery`).

Getting started
+++++++++++++++

pytorch nightly build should be installed, see
`Start Locally <https://pytorch.org/get-started/locally/>`_.

::

    git clone https://github.com/sdpython/experimental-experiment.git
    pip install onnxruntime-gpu nvidia-ml-py
    pip install -r requirements-dev.txt    
    export PYTHONPATH=$PYTHONPATH:<this folder>





.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Every conversion task must be tested on a large scale. One huge source of model is HuggingFace. We focus on the model Tiny-LLM. To avoid downloading any weigths, we write a function creating a random model based on the same architecture.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_exporter_exporter_untrained_tinyllm_thumb.png
    :alt:

  :doc:`/auto_recipes/plot_exporter_exporter_untrained_tinyllm`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Check the exporter on a dummy from HuggingFace</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="torch.export.export often breaks on big models because there are control flows or instructions breaking the propagation of dynamic shapes (see ...). The function usually gives an indication where the model implementation can be fixed but in case, that is not possible, we can try to export the model piece by piece: every module is converted separately from its submodule. A model can be exported even if one of its submodules cannot.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_exporter_exporter_phi35_piece_thumb.png
    :alt:

  :doc:`/auto_recipes/plot_exporter_exporter_phi35_piece`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Export Phi-3.5-mini-instruct piece by piece</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Tries torch.export._draft_export.draft_export.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_exporter_exporter_draft_mode_thumb.png
    :alt:

  :doc:`/auto_recipes/plot_exporter_exporter_draft_mode`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Export Phi-3.5-mini-instruct with draft_export</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Tries torch._export.tools.report_exportability.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_exporter_exporter_reportibility_thumb.png
    :alt:

  :doc:`/auto_recipes/plot_exporter_exporter_reportibility`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Export Phi-3.5-mini-instruct with report_exportability</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Control flow cannot be exported with a change. The code of the model can be changed or patched to introduce function torch.ops.higher_order.scan.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_exporter_exporter_scan_pdist_thumb.png
    :alt:

  :doc:`/auto_recipes/plot_exporter_exporter_scan_pdist`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Export a model with a loop (scan)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example shows how to convert a custom operator as defined in the tutorial Python Custom Operators.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_exporter_recipes_c_custom_ops_inplace_thumb.png
    :alt:

  :doc:`/auto_recipes/plot_exporter_recipes_c_custom_ops_inplace`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">to_onnx and a custom operator inplace</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example shows how to convert a custom operator, inspired from Python Custom Operators.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_exporter_recipes_c_custom_ops_fct_thumb.png
    :alt:

  :doc:`/auto_recipes/plot_exporter_recipes_c_custom_ops_fct`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">to_onnx and a custom operator registered with a function</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Control flow cannot be exported with a change. The code of the model can be changed or patched to introduce function torch.cond.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_exporter_recipes_c_cond_thumb.png
    :alt:

  :doc:`/auto_recipes/plot_exporter_recipes_c_cond`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">to_onnx and a model with a test</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This is a frequent task which does not play well with dynamic shapes. Let&#x27;s see how to avoid using torch.cond.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_exporter_recipes_c_dynpad_thumb.png
    :alt:

  :doc:`/auto_recipes/plot_exporter_recipes_c_dynpad`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">to_onnx and padding one dimension to a mulitple of a constant</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Big models are hard to read once converted into onnx. Let&#x27;s see how to improve their readibility. The code is inspired from LLM from scratch with Pytorch.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_exporter_recipes_c_modules_thumb.png
    :alt:

  :doc:`/auto_recipes/plot_exporter_recipes_c_modules`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">to_onnx and submodules from LLMs</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_recipes/plot_exporter_exporter_untrained_tinyllm
   /auto_recipes/plot_exporter_exporter_phi35_piece
   /auto_recipes/plot_exporter_exporter_draft_mode
   /auto_recipes/plot_exporter_exporter_reportibility
   /auto_recipes/plot_exporter_exporter_scan_pdist
   /auto_recipes/plot_exporter_recipes_c_custom_ops_inplace
   /auto_recipes/plot_exporter_recipes_c_custom_ops_fct
   /auto_recipes/plot_exporter_recipes_c_cond
   /auto_recipes/plot_exporter_recipes_c_dynpad
   /auto_recipes/plot_exporter_recipes_c_modules


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: auto_recipes_python.zip </auto_recipes/auto_recipes_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: auto_recipes_jupyter.zip </auto_recipes/auto_recipes_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
