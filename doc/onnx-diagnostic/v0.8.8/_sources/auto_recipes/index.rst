:orphan:

Common Export Issues
====================



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="torch.export.export does not work if a tensor given to the function has 0 or 1 for dimension declared as dynamic dimension.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_export_dim1_thumb.png
    :alt:

  :doc:`/auto_recipes/plot_export_dim1`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">0, 1, 2 for a Dynamic Dimension in the dummy example to export a model</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Getting dynamic shapes right for torch.export.export when the inputs includes a custom class such as a transformers.cache_utils.DynamicCache. torch.export.export cannot use a DynamicCache filled with dynamic shapes but instead it uses a kind of unserialized serialized form of it.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_dynamic_shapes_what_thumb.png
    :alt:

  :doc:`/auto_recipes/plot_dynamic_shapes_what`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Builds dynamic shapes from any input</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This is related to the following issues: Cannot export torch.sym_max(x.shape[0], y.shape[0]).">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_dynamic_shapes_max_thumb.png
    :alt:

  :doc:`/auto_recipes/plot_dynamic_shapes_max`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Cannot export torch.sym_max(x.shape[0], y.shape[0])</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="torch.export.export uses torch.SymInt to operate on shapes and optimizes the graph it produces. It checks if two tensors share the same dimension, if the shapes can be broadcast, ... To do that, python types must not be used or the algorithm looses information.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_dynamic_shapes_python_int_thumb.png
    :alt:

  :doc:`/auto_recipes/plot_dynamic_shapes_python_int`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Do not use python int with dynamic shapes</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Control flow cannot be exported with a change. The code of the model can be changed or patched to introduce function torch.cond.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_export_cond_thumb.png
    :alt:

  :doc:`/auto_recipes/plot_export_cond`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Export a model with a control flow (If)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="torch.nonzero returns the indices of the first zero found in a tensor. The output shape is unknown in the generic case but... If you have a 2D tensor with at least a nonzero value in every row, you can guess the dimension. But torch.export.export does not know what you know.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_dynamic_shapes_nonzero_thumb.png
    :alt:

  :doc:`/auto_recipes/plot_dynamic_shapes_nonzero`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Half certain nonzero</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Dynamic shapes given to torch.export.export must follow the same semantic. What if we confuse tuple and list when defining the dynamic shapes, how to restore the expected type assuming we know the inputs? Not often useful but maybe we will learn more about optree.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_dynamic_shapes_json_thumb.png
    :alt:

  :doc:`/auto_recipes/plot_dynamic_shapes_json`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">JSON returns list when the original dynamic shapes are list or tuple</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Setting the dynamic shapes is not always easy. Here are a few tricks to make it work.">

.. only:: html

  .. image:: /auto_recipes/images/thumb/sphx_glr_plot_export_with_dynamic_thumb.png
    :alt:

  :doc:`/auto_recipes/plot_export_with_dynamic`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Use DYNAMIC or AUTO when exporting if dynamic shapes has constraints</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_recipes/plot_export_dim1
   /auto_recipes/plot_dynamic_shapes_what
   /auto_recipes/plot_dynamic_shapes_max
   /auto_recipes/plot_dynamic_shapes_python_int
   /auto_recipes/plot_export_cond
   /auto_recipes/plot_dynamic_shapes_nonzero
   /auto_recipes/plot_dynamic_shapes_json
   /auto_recipes/plot_export_with_dynamic


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: auto_recipes_python.zip </auto_recipes/auto_recipes_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: auto_recipes_jupyter.zip </auto_recipes/auto_recipes_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
