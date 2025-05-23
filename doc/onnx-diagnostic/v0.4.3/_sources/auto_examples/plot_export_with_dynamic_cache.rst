
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_examples/plot_export_with_dynamic_cache.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_examples_plot_export_with_dynamic_cache.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_plot_export_with_dynamic_cache.py:


.. _l-plot-export-with-dynamic-shape:

===========================================
Export with DynamicCache and dynamic shapes
===========================================

Every LLMs implemented in :epkg:`transformers` use cache.
One of the most used is :class:`transformers.cache_utils.DynamicCache`.
The cache size is dynamic to cope with the growing context.
The example shows a tool which determines the dynamic shapes
for :func:`torch.export.export` based on a set of valid inputs.

DynamicCache
============

:func:`torch.export.export` serializes caches and any custom class
if these serialization functions are provided with is the case for
:class:`transformers.cache_utils.DynamicCache` and ``transformers>=4.50``.
The dynamic shapes must be provided following the serialized form.

.. GENERATED FROM PYTHON SOURCE LINES 22-60

.. code-block:: Python


    import pprint
    import torch
    from onnx_diagnostic import doc
    from onnx_diagnostic.ext_test_case import has_transformers
    from onnx_diagnostic.helpers import string_type
    from onnx_diagnostic.helpers.cache_helper import (
        flatten_unflatten_for_dynamic_shapes,
        make_dynamic_cache,
    )
    from onnx_diagnostic.export import ModelInputs
    from onnx_diagnostic.torch_export_patches import torch_export_patches


    class Model(torch.nn.Module):
        def forward(self, cache, z):
            return (
                z
                + cache.key_cache[0]
                + cache.key_cache[1]
                + cache.value_cache[0]
                + cache.value_cache[1]
            )


    model = Model()

    n_layers = 2
    bsize, nheads, slen, dim = 2, 4, 3, 7
    cache = make_dynamic_cache(
        [
            (torch.randn(bsize, nheads, slen, dim), torch.randn(bsize, nheads, slen, dim))
            for i in range(n_layers)
        ]
    )
    z = torch.randn((1, 1, 1, 7))
    model(cache, z)  # to check it works.





.. rst-class:: sphx-glr-script-out

 .. code-block:: none


    tensor([[[[-1.8605, -5.4504, -2.3892,  0.4026,  3.2462,  0.3760, -1.4181],
              [-2.8890, -2.2520, -0.2076, -2.6661,  1.3611,  2.2601, -1.8810],
              [-2.1088, -2.1647,  0.1484, -3.5379, -1.5968,  4.3554,  5.6420]],

             [[ 1.7319,  0.4264,  0.3296,  2.2837,  1.8617, -0.2268,  1.7536],
              [-1.1231,  0.1229, -2.4479,  1.7075,  4.6380, -0.5404, -0.0735],
              [-1.2506, -4.2822, -0.5056,  2.6407,  2.1057,  0.2186,  0.9482]],

             [[-3.1648, -0.1417,  3.6555, -2.5482,  1.3253,  0.4597,  0.5646],
              [-2.7349, -0.0772, -4.9611, -4.3342,  2.2564,  1.8305,  2.5631],
              [ 0.1012, -1.3093, -1.9298, -2.2986, -0.7864,  4.8788,  2.9099]],

             [[ 1.0965, -2.0789, -2.1207,  0.3917,  4.0795,  3.1778, -2.2771],
              [-4.8578, -6.9271, -6.5074,  2.1369, -0.8894, -0.1846, -0.8819],
              [-0.0309, -3.7392,  2.5612, -2.6582, -2.2330,  1.7601, -1.4208]]],


            [[[-0.3899, -2.6561, -5.8506, -3.9568, -0.4239,  1.0534,  6.2202],
              [-2.4387, -3.9891, -1.8072, -0.6490, -1.4709, -1.2247,  0.6174],
              [ 0.1999, -0.0763, -0.3712, -2.6219, -1.8042,  4.5284,  2.4134]],

             [[-1.7219, -3.4888, -1.7979, -5.6367,  1.3094,  0.8254,  0.6792],
              [ 0.4737,  4.8481, -0.5698, -2.6678, -0.3776,  0.8704,  2.4824],
              [-2.5786, -1.0099,  2.0233,  3.8570,  4.4071,  0.5649,  2.2431]],

             [[ 2.2775, -2.1189, -1.5625,  2.7245,  1.0061,  0.2306, -0.2830],
              [-1.0669, -1.2480, -3.3949, -0.2680, -1.2898,  0.5371,  1.0551],
              [ 2.7892, -3.7749,  0.1293, -1.5835, -1.5676, -1.1500, -0.6408]],

             [[ 2.3852,  1.0136,  1.2422, -0.0792,  2.0969,  0.6596, -2.9321],
              [-1.7794,  1.1378, -0.9360, -3.3211, -0.4293,  1.4050,  2.0528],
              [-2.3679, -0.9945,  1.0518, -4.2388, -1.1410,  2.0224, -2.4178]]]])



.. GENERATED FROM PYTHON SOURCE LINES 61-62

The cache looks like this:

.. GENERATED FROM PYTHON SOURCE LINES 62-66

.. code-block:: Python


    print(string_type(cache, with_shape=True))






.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    DynamicCache(key_cache=#2[T1s2x4x3x7,T1s2x4x3x7], value_cache=#2[T1s2x4x3x7,T1s2x4x3x7])




.. GENERATED FROM PYTHON SOURCE LINES 67-82

.. code-block:: Python


    cache2 = make_dynamic_cache(
        [
            (
                torch.randn(bsize + 1, nheads, slen + 1, dim + 1),
                torch.randn(bsize + 1, nheads, slen + 1, dim + 1),
            )
            for i in range(n_layers)
        ]
    )
    inputs = [
        (cache, z),
        (cache2, torch.randn((1, 1, 1, 8))),
    ]








.. GENERATED FROM PYTHON SOURCE LINES 83-84

And the second set of inputs looks like:

.. GENERATED FROM PYTHON SOURCE LINES 84-86

.. code-block:: Python

    print(string_type(inputs[1], with_shape=True))





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    (DynamicCache(key_cache=#2[T1s3x4x4x8,T1s3x4x4x8], value_cache=#2[T1s3x4x4x8,T1s3x4x4x8]),T1s1x1x1x8)




.. GENERATED FROM PYTHON SOURCE LINES 87-92

Guess the dynamic shapes
========================

The following tool can be used to guess the dynamic shapes
the way :func:`torch.export.export` expects them.

.. GENERATED FROM PYTHON SOURCE LINES 92-98

.. code-block:: Python


    mi = ModelInputs(Model(), inputs)
    ds = mi.guess_dynamic_shapes()

    pprint.pprint(ds)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    (([[{0: _DimHint(type=<_DimHintType.DYNAMIC: 3>,
                     min=None,
                     max=None,
                     _factory=True),
         2: _DimHint(type=<_DimHintType.DYNAMIC: 3>,
                     min=None,
                     max=None,
                     _factory=True),
         3: _DimHint(type=<_DimHintType.DYNAMIC: 3>,
                     min=None,
                     max=None,
                     _factory=True)},
        {0: _DimHint(type=<_DimHintType.DYNAMIC: 3>,
                     min=None,
                     max=None,
                     _factory=True),
         2: _DimHint(type=<_DimHintType.DYNAMIC: 3>,
                     min=None,
                     max=None,
                     _factory=True),
         3: _DimHint(type=<_DimHintType.DYNAMIC: 3>,
                     min=None,
                     max=None,
                     _factory=True)}],
       [{0: _DimHint(type=<_DimHintType.DYNAMIC: 3>,
                     min=None,
                     max=None,
                     _factory=True),
         2: _DimHint(type=<_DimHintType.DYNAMIC: 3>,
                     min=None,
                     max=None,
                     _factory=True),
         3: _DimHint(type=<_DimHintType.DYNAMIC: 3>,
                     min=None,
                     max=None,
                     _factory=True)},
        {0: _DimHint(type=<_DimHintType.DYNAMIC: 3>,
                     min=None,
                     max=None,
                     _factory=True),
         2: _DimHint(type=<_DimHintType.DYNAMIC: 3>,
                     min=None,
                     max=None,
                     _factory=True),
         3: _DimHint(type=<_DimHintType.DYNAMIC: 3>,
                     min=None,
                     max=None,
                     _factory=True)}]],
      {3: _DimHint(type=<_DimHintType.DYNAMIC: 3>,
                   min=None,
                   max=None,
                   _factory=True)}),
     {})




.. GENERATED FROM PYTHON SOURCE LINES 99-105

And finally the export.
The export is simple if ``transformers>=4.50``, otherwise,
transformers needs to be patched.
:func:`onnx_diagnostic.torch_export_patches.torch_export_patches`
registers functions to serialize ``DynamicCache``. This one is modified to make
the shape inference implemented in :epkg:`torch` happy.

.. GENERATED FROM PYTHON SOURCE LINES 105-124

.. code-block:: Python


    if has_transformers("4.50"):
        ep = torch.export.export(model, inputs[0], dynamic_shapes=ds[0], strict=False)
    else:
        with torch_export_patches(patch_transformers=True) as modificator:
            ep = torch.export.export(
                model, modificator(inputs[0]), dynamic_shapes=ds[0], strict=False
            )
    print(ep)

    # Do we need to guess?
    # ++++++++++++++++++++
    #
    # Function :func:`onnx_diagnostic.helpers.string_type` is using
    # the serialization functions to print out the DynamicCache the was
    # :func:`torch.export.export` expects them.

    print(string_type(cache, with_shape=True))





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    ExportedProgram:
        class GraphModule(torch.nn.Module):
            def forward(self, cache_key_cache_0: "f32[s26, 4, s28, s1]", cache_key_cache_1: "f32[s26, 4, s28, s1]", cache_value_cache_0: "f32[s26, 4, s28, s1]", cache_value_cache_1: "f32[s26, 4, s28, s1]", z: "f32[1, 1, 1, s1]"):
                 # File: /home/xadupre/github/onnx-diagnostic/_doc/examples/plot_export_with_dynamic_cache.py:39 in forward, code: z
                add: "f32[s26, 4, s28, s1]" = torch.ops.aten.add.Tensor(z, cache_key_cache_0);  z = cache_key_cache_0 = None
                add_1: "f32[s26, 4, s28, s1]" = torch.ops.aten.add.Tensor(add, cache_key_cache_1);  add = cache_key_cache_1 = None
                add_2: "f32[s26, 4, s28, s1]" = torch.ops.aten.add.Tensor(add_1, cache_value_cache_0);  add_1 = cache_value_cache_0 = None
                add_3: "f32[s26, 4, s28, s1]" = torch.ops.aten.add.Tensor(add_2, cache_value_cache_1);  add_2 = cache_value_cache_1 = None
                return (add_3,)
            
    Graph signature: 
        # inputs
        cache_key_cache_0: USER_INPUT
        cache_key_cache_1: USER_INPUT
        cache_value_cache_0: USER_INPUT
        cache_value_cache_1: USER_INPUT
        z: USER_INPUT
    
        # outputs
        add_3: USER_OUTPUT
    
    Range constraints: {s26: VR[2, int_oo], s28: VR[2, int_oo], s1: VR[2, int_oo]}

    DynamicCache(key_cache=#2[T1s2x4x3x7,T1s2x4x3x7], value_cache=#2[T1s2x4x3x7,T1s2x4x3x7])




.. GENERATED FROM PYTHON SOURCE LINES 125-129

You can also use function
:func:`onnx_diagnostic.helpers.cache_helper.flatten_unflatten_for_dynamic_shapes`
to show a DynamicCache restructured the way :func:`torch.export.export` expects
it to be without the custom class.

.. GENERATED FROM PYTHON SOURCE LINES 129-132

.. code-block:: Python


    print(string_type(flatten_unflatten_for_dynamic_shapes(cache), with_shape=True))





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    #2[#2[T1s2x4x3x7,T1s2x4x3x7],#2[T1s2x4x3x7,T1s2x4x3x7]]




.. GENERATED FROM PYTHON SOURCE LINES 133-135

This code works for any custom class if it was registered
with :func:`torch.utils._pytree.register_pytree_node`.

.. GENERATED FROM PYTHON SOURCE LINES 135-138

.. code-block:: Python



    doc.plot_legend("dynamic shapes\nfor DynamicCache", "torch.export.export", "tomato")



.. image-sg:: /auto_examples/images/sphx_glr_plot_export_with_dynamic_cache_001.png
   :alt: plot export with dynamic cache
   :srcset: /auto_examples/images/sphx_glr_plot_export_with_dynamic_cache_001.png
   :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 0.310 seconds)


.. _sphx_glr_download_auto_examples_plot_export_with_dynamic_cache.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_export_with_dynamic_cache.ipynb <plot_export_with_dynamic_cache.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_export_with_dynamic_cache.py <plot_export_with_dynamic_cache.py>`

    .. container:: sphx-glr-download sphx-glr-download-zip

      :download:`Download zipped: plot_export_with_dynamic_cache.zip <plot_export_with_dynamic_cache.zip>`


.. include:: plot_export_with_dynamic_cache.recommendations


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
