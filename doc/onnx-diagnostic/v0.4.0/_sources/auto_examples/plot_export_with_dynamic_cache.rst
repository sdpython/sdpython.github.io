
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
    from onnx_diagnostic.torch_export_patches import bypass_export_some_errors


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


    tensor([[[[ 2.0641,  1.8414,  2.1144, -2.1858, -1.4553,  2.1089, -3.9414],
              [ 2.5480,  0.4189,  0.8150, -1.8488, -1.5655,  1.6026, -0.3057],
              [ 3.8920, -1.8116,  0.7533, -2.1390,  0.9199, -0.8702,  2.0602]],

             [[ 0.4179, -0.7373,  0.9346,  1.8390, -3.5611,  3.4647,  0.2530],
              [-2.0087,  1.1262, -1.9001,  3.3054, -2.2922,  2.4464,  0.9105],
              [ 1.6731,  0.8197, -0.3959,  4.5894, -1.9321, -2.5554, -0.1521]],

             [[-0.4913, -1.6713, -0.8794,  2.4646, -1.6725, -3.4529, -2.7974],
              [ 4.2529, -4.1975,  0.4804, -1.7241, -2.2581, -1.1349, -1.2331],
              [ 3.0714, -0.8710, -4.8588,  3.2943, -2.8378, -0.1089, -1.8729]],

             [[ 1.0005,  3.5533, -2.5484, -1.0387, -1.3163,  1.9498, -1.7814],
              [-1.0856, -2.3275,  3.0730, -0.1047, -3.7895,  1.5157,  0.0685],
              [ 0.7635,  1.6033,  1.8651, -0.7784, -1.7362, -1.0176,  0.9097]]],


            [[[ 0.4167, -3.1517,  0.4268, -0.6477, -2.1251, -2.2218, -4.0205],
              [-0.3417, -0.5707, -1.0588, -0.4271, -0.6683, -1.1274, -3.2400],
              [ 6.3509, -0.7531,  1.9027,  1.8726, -4.0646,  0.4263,  2.2371]],

             [[ 0.2658,  3.4582, -4.4293,  1.3780, -2.8750, -1.7799,  0.5450],
              [ 0.7751, -1.5647,  0.0314,  3.6410, -2.6702,  0.5702,  2.2426],
              [-0.6385, -0.9095,  1.5091, -1.2678, -3.0067, -0.2964,  1.6353]],

             [[ 1.9896,  2.1162,  2.8666,  0.4280, -2.3102, -1.7958,  0.4322],
              [ 3.9568,  0.4840, -2.0115,  0.9279, -2.8587,  0.3561, -0.0295],
              [ 0.6160,  0.1291,  0.5133, -2.9567, -2.2645,  0.7163,  3.2457]],

             [[ 0.8378,  0.2365,  0.0756, -0.3337, -2.9120, -1.5043, -0.7300],
              [-0.2882,  1.1744, -3.2432, -3.2175, -0.3126, -0.3074, -1.8022],
              [ 0.7445, -0.6921, -1.7870,  2.7280, -6.2481,  4.1910,  0.4509]]]])



.. GENERATED FROM PYTHON SOURCE LINES 61-62

The cache looks like this:

.. GENERATED FROM PYTHON SOURCE LINES 62-66

.. code-block:: Python


    print(string_type(cache, with_shape=True))






.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    DynamicCache[serialized](#2[#2[T1s2x4x3x7,T1s2x4x3x7],#2[T1s2x4x3x7,T1s2x4x3x7]])




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

    (DynamicCache[serialized](#2[#2[T1s3x4x4x8,T1s3x4x4x8],#2[T1s3x4x4x8,T1s3x4x4x8]]),T1s1x1x1x8)




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
:func:`onnx_diagnostic.torch_export_patches.bypass_export_some_errors`
registers functions to serialize ``DynamicCache``. This one is modified to make
the shape inference implemented in :epkg:`torch` happy.

.. GENERATED FROM PYTHON SOURCE LINES 105-124

.. code-block:: Python


    if has_transformers("4.50"):
        ep = torch.export.export(model, inputs[0], dynamic_shapes=ds[0], strict=False)
    else:
        with bypass_export_some_errors(patch_transformers=True) as modificator:
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

    DynamicCache[serialized](#2[#2[T1s2x4x3x7,T1s2x4x3x7],#2[T1s2x4x3x7,T1s2x4x3x7]])




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

   **Total running time of the script:** (0 minutes 0.197 seconds)


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
