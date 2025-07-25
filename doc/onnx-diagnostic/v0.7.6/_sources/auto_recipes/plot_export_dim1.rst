
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_recipes/plot_export_dim1.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_recipes_plot_export_dim1.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_recipes_plot_export_dim1.py:


.. _l-plot-export-dim1:

0, 1, 2 for a Dynamic Dimension in the dummy example to export a model
======================================================================

:func:`torch.export.export` does not work if a tensor given to the function
has 0 or 1 for dimension declared as dynamic dimension.

Simple model, no dimension with 0 or 1
++++++++++++++++++++++++++++++++++++++

.. GENERATED FROM PYTHON SOURCE LINES 13-35

.. code-block:: Python


    import torch
    from onnx_diagnostic import doc


    class Model(torch.nn.Module):
        def forward(self, x, y, z):
            return torch.cat((x, y), axis=1) + z


    model = Model()
    x = torch.randn(2, 3)
    y = torch.randn(2, 5)
    z = torch.randn(2, 8)
    model(x, y, z)

    DYN = torch.export.Dim.DYNAMIC
    ds = {0: DYN, 1: DYN}

    ep = torch.export.export(model, (x, y, z), dynamic_shapes=(ds, ds, ds))
    print(ep.graph)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    graph():
        %x : [num_users=3] = placeholder[target=x]
        %y : [num_users=3] = placeholder[target=y]
        %z : [num_users=3] = placeholder[target=z]
        %sym_size_int : [num_users=1] = call_function[target=torch.ops.aten.sym_size.int](args = (%x, 0), kwargs = {})
        %sym_size_int_1 : [num_users=1] = call_function[target=torch.ops.aten.sym_size.int](args = (%x, 1), kwargs = {})
        %sym_size_int_2 : [num_users=2] = call_function[target=torch.ops.aten.sym_size.int](args = (%y, 0), kwargs = {})
        %sym_size_int_3 : [num_users=1] = call_function[target=torch.ops.aten.sym_size.int](args = (%y, 1), kwargs = {})
        %sym_size_int_4 : [num_users=1] = call_function[target=torch.ops.aten.sym_size.int](args = (%z, 0), kwargs = {})
        %sym_size_int_5 : [num_users=1] = call_function[target=torch.ops.aten.sym_size.int](args = (%z, 1), kwargs = {})
        %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%x, %y], 1), kwargs = {})
        %eq : [num_users=1] = call_function[target=operator.eq](args = (%sym_size_int_2, %sym_size_int), kwargs = {})
        %_assert_scalar_default : [num_users=0] = call_function[target=torch.ops.aten._assert_scalar.default](args = (%eq, Runtime assertion failed for expression Eq(s17, s77) on node 'eq'), kwargs = {})
        %add_1 : [num_users=1] = call_function[target=operator.add](args = (%sym_size_int_1, %sym_size_int_3), kwargs = {})
        %eq_1 : [num_users=1] = call_function[target=operator.eq](args = (%add_1, %sym_size_int_5), kwargs = {})
        %_assert_scalar_default_1 : [num_users=0] = call_function[target=torch.ops.aten._assert_scalar.default](args = (%eq_1, Runtime assertion failed for expression Eq(s27 + s94, s32) on node 'eq_1'), kwargs = {})
        %eq_2 : [num_users=1] = call_function[target=operator.eq](args = (%sym_size_int_2, %sym_size_int_4), kwargs = {})
        %_assert_scalar_default_2 : [num_users=0] = call_function[target=torch.ops.aten._assert_scalar.default](args = (%eq_2, Runtime assertion failed for expression Eq(s17, s68) on node 'eq_2'), kwargs = {})
        %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat, %z), kwargs = {})
        return (add,)




.. GENERATED FROM PYTHON SOURCE LINES 36-38

Same model, a dynamic dimension = 1
+++++++++++++++++++++++++++++++++++

.. GENERATED FROM PYTHON SOURCE LINES 38-50

.. code-block:: Python


    z = z[:1]

    DYN = torch.export.Dim.DYNAMIC
    ds = {0: DYN, 1: DYN}

    try:
        ep = torch.export.export(model, (x, y, z), dynamic_shapes=(ds, ds, ds))
        print(ep.graph)
    except Exception as e:
        print("ERROR", e)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    ERROR Found the following conflicts between user-specified ranges and inferred ranges from model tracing:
    - Received user-specified dim hint Dim.DYNAMIC(min=None, max=None), but export 0/1 specialized due to hint of 1 for dimension inputs['z'].shape[0].




.. GENERATED FROM PYTHON SOURCE LINES 51-52

It failed. Let's try a little trick.

.. GENERATED FROM PYTHON SOURCE LINES 54-56

Same model, a dynamic dimension = 1 and backed_size_oblivious=True
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. GENERATED FROM PYTHON SOURCE LINES 56-61

.. code-block:: Python


    with torch.fx.experimental._config.patch(backed_size_oblivious=True):
        ep = torch.export.export(model, (x, y, z), dynamic_shapes=(ds, ds, ds))
        print(ep.graph)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    graph():
        %x : [num_users=3] = placeholder[target=x]
        %y : [num_users=3] = placeholder[target=y]
        %z : [num_users=3] = placeholder[target=z]
        %sym_size_int : [num_users=1] = call_function[target=torch.ops.aten.sym_size.int](args = (%x, 0), kwargs = {})
        %sym_size_int_1 : [num_users=1] = call_function[target=torch.ops.aten.sym_size.int](args = (%x, 1), kwargs = {})
        %sym_size_int_2 : [num_users=2] = call_function[target=torch.ops.aten.sym_size.int](args = (%y, 0), kwargs = {})
        %sym_size_int_3 : [num_users=1] = call_function[target=torch.ops.aten.sym_size.int](args = (%y, 1), kwargs = {})
        %sym_size_int_4 : [num_users=1] = call_function[target=torch.ops.aten.sym_size.int](args = (%z, 0), kwargs = {})
        %sym_size_int_5 : [num_users=1] = call_function[target=torch.ops.aten.sym_size.int](args = (%z, 1), kwargs = {})
        %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%x, %y], 1), kwargs = {})
        %eq : [num_users=1] = call_function[target=operator.eq](args = (%sym_size_int_2, %sym_size_int), kwargs = {})
        %_assert_scalar_default : [num_users=0] = call_function[target=torch.ops.aten._assert_scalar.default](args = (%eq, Runtime assertion failed for expression Eq(s17, s77) on node 'eq'), kwargs = {})
        %add_1 : [num_users=1] = call_function[target=operator.add](args = (%sym_size_int_1, %sym_size_int_3), kwargs = {})
        %eq_1 : [num_users=1] = call_function[target=operator.eq](args = (%add_1, %sym_size_int_5), kwargs = {})
        %_assert_scalar_default_1 : [num_users=0] = call_function[target=torch.ops.aten._assert_scalar.default](args = (%eq_1, Runtime assertion failed for expression Eq(s27 + s94, s32) on node 'eq_1'), kwargs = {})
        %eq_2 : [num_users=1] = call_function[target=operator.eq](args = (%sym_size_int_2, %sym_size_int_4), kwargs = {})
        %_assert_scalar_default_2 : [num_users=0] = call_function[target=torch.ops.aten._assert_scalar.default](args = (%eq_2, Runtime assertion failed for expression Eq(s17, s68) on node 'eq_2'), kwargs = {})
        %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat, %z), kwargs = {})
        return (add,)




.. GENERATED FROM PYTHON SOURCE LINES 62-63

It worked.

.. GENERATED FROM PYTHON SOURCE LINES 63-65

.. code-block:: Python


    doc.plot_legend("dynamic dimension\nworking with\n0 or 1", "torch.export.export", "green")



.. image-sg:: /auto_recipes/images/sphx_glr_plot_export_dim1_001.png
   :alt: plot export dim1
   :srcset: /auto_recipes/images/sphx_glr_plot_export_dim1_001.png
   :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 0.470 seconds)


.. _sphx_glr_download_auto_recipes_plot_export_dim1.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_export_dim1.ipynb <plot_export_dim1.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_export_dim1.py <plot_export_dim1.py>`

    .. container:: sphx-glr-download sphx-glr-download-zip

      :download:`Download zipped: plot_export_dim1.zip <plot_export_dim1.zip>`


.. include:: plot_export_dim1.recommendations


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
