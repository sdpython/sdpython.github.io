
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

Simple Examples
===============

We first look at examples playing positional and names parameters
to understand how :func:`torch.export.export` works.

args
++++

.. GENERATED FROM PYTHON SOURCE LINES 23-31

.. code-block:: Python


    import pprint
    import torch
    from onnx_diagnostic import doc
    from onnx_diagnostic.helpers.cache_helper import make_dynamic_cache
    from onnx_diagnostic.helpers import string_type
    from onnx_diagnostic.export import ModelInputs








.. GENERATED FROM PYTHON SOURCE LINES 32-34

We need addition import in case ``transformers<4.50``.
Exporting DynamicCache is not supported before that.

.. GENERATED FROM PYTHON SOURCE LINES 34-51

.. code-block:: Python

    from onnx_diagnostic.ext_test_case import has_transformers
    from onnx_diagnostic.torch_export_patches import bypass_export_some_errors


    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x + y


    model = Model()
    x = torch.randn((5, 6))
    y = torch.randn((1, 6))
    model(x, y)  # to check it works

    ep = torch.export.export(model, (x, y))
    print(ep)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    /home/xadupre/vv/this312/lib/python3.12/site-packages/torch/backends/mkldnn/__init__.py:78: UserWarning: TF32 acceleration on top of oneDNN is available for Intel GPUs. The current Torch version does not have Intel GPU Support. (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:148.)
      torch._C._set_onednn_allow_tf32(_allow_tf32)
    ExportedProgram:
        class GraphModule(torch.nn.Module):
            def forward(self, x: "f32[5, 6]", y: "f32[1, 6]"):
                 # File: /home/xadupre/github/onnx-diagnostic/_doc/examples/plot_export_with_dynamic_cache.py:40 in forward, code: return x + y
                add: "f32[5, 6]" = torch.ops.aten.add.Tensor(x, y);  x = y = None
                return (add,)
            
    Graph signature: 
        # inputs
        x: USER_INPUT
        y: USER_INPUT
    
        # outputs
        add: USER_OUTPUT
    
    Range constraints: {}





.. GENERATED FROM PYTHON SOURCE LINES 52-57

As expected there is no dynamic shapes.
We use :class:`onnx_diagnostic.export.ModelInputs`
to define them from two set of valid inputs.
These inputs must have different value for the dynamic
dimensions.

.. GENERATED FROM PYTHON SOURCE LINES 57-63

.. code-block:: Python


    inputs = [(x, y), (torch.randn((7, 8)), torch.randn((1, 8)))]
    mi = ModelInputs(Model(), inputs)
    ds = mi.guess_dynamic_shapes()
    pprint.pprint(ds)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    (({0: _DimHint(type=<_DimHintType.DYNAMIC: 3>),
       1: _DimHint(type=<_DimHintType.DYNAMIC: 3>)},
      {1: _DimHint(type=<_DimHintType.DYNAMIC: 3>)}),
     {})




.. GENERATED FROM PYTHON SOURCE LINES 64-68

The function returns a tuple with two objects.
The first one for the positional arguments, the other one
for the named arguments. There is no named arguments. We
we used the first result to export.

.. GENERATED FROM PYTHON SOURCE LINES 68-72

.. code-block:: Python


    ep = torch.export.export(model, (x, y), dynamic_shapes=ds[0])
    print(ep)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    /home/xadupre/vv/this312/lib/python3.12/site-packages/torch/backends/mkldnn/__init__.py:78: UserWarning: TF32 acceleration on top of oneDNN is available for Intel GPUs. The current Torch version does not have Intel GPU Support. (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:148.)
      torch._C._set_onednn_allow_tf32(_allow_tf32)
    ExportedProgram:
        class GraphModule(torch.nn.Module):
            def forward(self, x: "f32[s0, s1]", y: "f32[1, s1]"):
                 # File: /home/xadupre/github/onnx-diagnostic/_doc/examples/plot_export_with_dynamic_cache.py:40 in forward, code: return x + y
                add: "f32[s0, s1]" = torch.ops.aten.add.Tensor(x, y);  x = y = None
                return (add,)
            
    Graph signature: 
        # inputs
        x: USER_INPUT
        y: USER_INPUT
    
        # outputs
        add: USER_OUTPUT
    
    Range constraints: {s0: VR[2, int_oo], s1: VR[2, int_oo]}





.. GENERATED FROM PYTHON SOURCE LINES 73-77

kwargs
++++++

We do the same with named arguments.

.. GENERATED FROM PYTHON SOURCE LINES 77-89

.. code-block:: Python



    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x + y


    model = Model()
    x = torch.randn((5, 6))
    y = torch.randn((1, 6))
    model(x=x, y=y)  # to check it works





.. rst-class:: sphx-glr-script-out

 .. code-block:: none


    tensor([[-0.1448, -0.1862,  1.1237,  0.6545, -0.4959, -0.0870],
            [ 0.2349, -0.5582,  1.8353,  1.1814,  0.1543, -1.9090],
            [-3.2581,  0.8515,  0.5275,  1.2872, -1.4050,  1.0963],
            [ 1.0549,  1.5685,  1.4807,  0.8272, -3.0529, -0.7287],
            [ 0.2411,  2.1068,  0.8151,  0.5721, -0.2062, -1.4457]])



.. GENERATED FROM PYTHON SOURCE LINES 90-91

Two sets of valid inputs.

.. GENERATED FROM PYTHON SOURCE LINES 91-96

.. code-block:: Python

    inputs = [dict(x=x, y=y), dict(x=torch.randn((7, 8)), y=torch.randn((1, 8)))]
    mi = ModelInputs(Model(), inputs)
    ds = mi.guess_dynamic_shapes()
    pprint.pprint(ds)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    ((),
     {'x': {0: _DimHint(type=<_DimHintType.DYNAMIC: 3>),
            1: _DimHint(type=<_DimHintType.DYNAMIC: 3>)},
      'y': {1: _DimHint(type=<_DimHintType.DYNAMIC: 3>)}})




.. GENERATED FROM PYTHON SOURCE LINES 97-98

And we export.

.. GENERATED FROM PYTHON SOURCE LINES 98-101

.. code-block:: Python

    ep = torch.export.export(model, (), kwargs=dict(x=x, y=y), dynamic_shapes=ds[1])
    print(ep)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    /home/xadupre/vv/this312/lib/python3.12/site-packages/torch/backends/mkldnn/__init__.py:78: UserWarning: TF32 acceleration on top of oneDNN is available for Intel GPUs. The current Torch version does not have Intel GPU Support. (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:148.)
      torch._C._set_onednn_allow_tf32(_allow_tf32)
    ExportedProgram:
        class GraphModule(torch.nn.Module):
            def forward(self, x: "f32[s0, s1]", y: "f32[1, s1]"):
                 # File: /home/xadupre/github/onnx-diagnostic/_doc/examples/plot_export_with_dynamic_cache.py:81 in forward, code: return x + y
                add: "f32[s0, s1]" = torch.ops.aten.add.Tensor(x, y);  x = y = None
                return (add,)
            
    Graph signature: 
        # inputs
        x: USER_INPUT
        y: USER_INPUT
    
        # outputs
        add: USER_OUTPUT
    
    Range constraints: {s0: VR[2, int_oo], s1: VR[2, int_oo]}





.. GENERATED FROM PYTHON SOURCE LINES 102-107

args and kwargs
+++++++++++++++

:func:`torch.export.export` does not like having dynami shapes
for both args and kwargs. We need to define them using one mechanism.

.. GENERATED FROM PYTHON SOURCE LINES 107-119

.. code-block:: Python



    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x + y


    model = Model()
    x = torch.randn((5, 6))
    y = torch.randn((1, 6))
    model(x, y=y)  # to check it works





.. rst-class:: sphx-glr-script-out

 .. code-block:: none


    tensor([[ 0.4204,  1.6478, -0.8171,  0.1855, -1.9409, -0.5327],
            [-0.4473,  2.7178, -0.2275,  2.0488,  0.6332,  0.0117],
            [ 0.3611,  1.9432,  0.2684,  1.4006,  0.0700, -0.4519],
            [ 0.4502,  2.3819, -0.1065,  1.4964, -1.9907, -0.5859],
            [ 1.8839,  0.5022,  2.1120, -0.9325, -1.1525,  1.0515]])



.. GENERATED FROM PYTHON SOURCE LINES 120-121

Two sets of valid inputs with positional and names arguments.

.. GENERATED FROM PYTHON SOURCE LINES 121-127

.. code-block:: Python


    inputs = [((x,), dict(y=y)), ((torch.randn((7, 8)),), dict(y=torch.randn((1, 8))))]
    mi = ModelInputs(Model(), inputs)
    ds = mi.guess_dynamic_shapes()
    pprint.pprint(ds)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    (({0: _DimHint(type=<_DimHintType.DYNAMIC: 3>),
       1: _DimHint(type=<_DimHintType.DYNAMIC: 3>)},),
     {'y': {1: _DimHint(type=<_DimHintType.DYNAMIC: 3>)}})




.. GENERATED FROM PYTHON SOURCE LINES 128-132

This does not work with :func:`torch.export.export` so
we use a method to move the positional dynamic shapes to
named one. The method relies on the signature of the
forward method.

.. GENERATED FROM PYTHON SOURCE LINES 132-136

.. code-block:: Python


    new_args, new_kwargs, new_ds = mi.move_to_kwargs(*mi.inputs[0], ds)
    pprint.pprint(new_ds)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    ((),
     {'x': {0: _DimHint(type=<_DimHintType.DYNAMIC: 3>),
            1: _DimHint(type=<_DimHintType.DYNAMIC: 3>)},
      'y': {1: _DimHint(type=<_DimHintType.DYNAMIC: 3>)}})




.. GENERATED FROM PYTHON SOURCE LINES 137-138

And we export.

.. GENERATED FROM PYTHON SOURCE LINES 138-142

.. code-block:: Python


    ep = torch.export.export(model, new_args, kwargs=new_kwargs, dynamic_shapes=new_ds[1])
    print(ep)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    /home/xadupre/vv/this312/lib/python3.12/site-packages/torch/backends/mkldnn/__init__.py:78: UserWarning: TF32 acceleration on top of oneDNN is available for Intel GPUs. The current Torch version does not have Intel GPU Support. (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:148.)
      torch._C._set_onednn_allow_tf32(_allow_tf32)
    ExportedProgram:
        class GraphModule(torch.nn.Module):
            def forward(self, x: "f32[s0, s1]", y: "f32[1, s1]"):
                 # File: /home/xadupre/github/onnx-diagnostic/_doc/examples/plot_export_with_dynamic_cache.py:111 in forward, code: return x + y
                add: "f32[s0, s1]" = torch.ops.aten.add.Tensor(x, y);  x = y = None
                return (add,)
            
    Graph signature: 
        # inputs
        x: USER_INPUT
        y: USER_INPUT
    
        # outputs
        add: USER_OUTPUT
    
    Range constraints: {s0: VR[2, int_oo], s1: VR[2, int_oo]}





.. GENERATED FROM PYTHON SOURCE LINES 143-150

DynamicCache
============

:func:`torch.export.export` serializes caches and any custom class
if these serialization functions are provided with is the case for
:class:`transformers.cache_utils.DynamicCache` and ``transformers>=4.50``.
The dynamic shapes must be provided following the serialized form.

.. GENERATED FROM PYTHON SOURCE LINES 150-176

.. code-block:: Python



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


    tensor([[[[-3.8678e+00, -3.2331e+00, -3.8153e+00,  4.1202e-01,  2.1248e+00,
                1.6443e+00,  4.9690e-01],
              [-1.8551e+00,  3.5224e+00,  2.7558e-01,  3.3791e+00,  2.6265e+00,
                3.7658e+00, -7.5243e-01],
              [ 4.2960e-01,  3.1940e+00, -1.3051e+00,  2.2574e+00,  3.1201e+00,
                1.7933e+00, -4.1255e-01]],

             [[ 2.3725e+00, -1.9770e+00,  1.7464e+00,  1.0566e+00,  1.2862e+00,
                1.9009e-01,  5.9562e-01],
              [ 2.5476e+00, -1.5471e+00, -1.0872e+00,  1.9455e+00, -2.0705e+00,
                1.1152e+00, -2.0919e+00],
              [ 3.5627e+00, -5.2837e+00, -4.9835e-01,  1.5270e+00, -3.9628e+00,
               -1.1904e-02, -1.7063e+00]],

             [[-4.1200e-01, -2.3686e+00,  4.2592e-01, -5.0301e-01, -2.7219e-01,
               -1.0548e+00, -7.3917e-01],
              [-1.9590e+00, -1.4849e-01, -3.0877e+00, -3.2284e-01, -2.2212e+00,
                2.5695e-01, -5.0390e+00],
              [-2.8251e+00,  2.0815e-01, -3.7958e+00, -5.6984e-01, -1.3169e-03,
                2.4199e+00,  2.1275e+00]],

             [[ 5.7662e+00,  1.5418e+00, -2.7520e+00,  4.8957e-01,  7.2408e-01,
                3.4678e+00, -1.3289e+00],
              [-2.7180e+00, -2.1783e+00,  4.7699e+00,  4.2232e+00, -1.4109e+00,
                1.2840e+00, -3.1049e+00],
              [-8.4773e-01,  3.9937e+00, -3.5702e+00, -1.4239e+00,  5.7208e-02,
               -2.9228e-01,  1.7354e+00]]],


            [[[-1.2487e+00, -1.3192e-01, -2.7859e+00, -1.9405e+00, -2.4859e+00,
                1.9401e+00, -5.5388e-01],
              [ 1.5075e+00, -4.5329e+00, -9.6758e-02, -7.9823e-01, -4.0910e+00,
               -2.1267e+00,  2.4688e+00],
              [ 1.8432e+00, -3.0507e+00, -1.8005e+00,  2.4020e+00,  2.4400e+00,
                5.6654e-02,  1.0270e+00]],

             [[ 1.3554e+00, -6.8065e-01, -1.9378e+00,  6.5455e-01, -8.5219e-01,
                5.5111e-01, -9.8301e-02],
              [ 2.6432e+00, -2.2546e+00,  1.8647e+00,  3.6664e+00, -7.7408e-02,
                2.7038e+00, -1.5446e+00],
              [-3.3415e+00,  3.8830e+00, -3.4530e+00,  9.9113e-01,  5.0211e-01,
                1.4624e+00, -3.4173e-01]],

             [[-1.7096e-01, -7.9618e-01, -3.0700e+00,  1.1085e+00,  2.3456e+00,
               -1.1427e+00, -2.1442e+00],
              [-9.7848e-01, -6.8107e-01, -2.4141e-01, -1.6232e+00,  3.9734e+00,
                2.8699e-01, -2.8845e+00],
              [-7.3560e-01, -5.6911e-01,  1.1421e+00,  8.9931e-01,  1.0583e+00,
               -1.9487e+00,  1.7527e+00]],

             [[ 3.7169e+00,  1.2655e+00, -1.5434e+00,  2.6236e+00,  4.6914e+00,
               -3.7603e-01, -3.3455e+00],
              [-1.0418e+00,  4.8539e-01,  1.1190e+00,  2.7838e+00, -5.5089e-01,
               -1.3996e-01, -1.7486e+00],
              [-2.4874e+00, -3.9947e+00, -3.4697e-02,  1.0953e+00,  5.8200e-02,
               -1.4247e+00, -3.4249e-01]]]])



.. GENERATED FROM PYTHON SOURCE LINES 177-178

The cache looks like this:

.. GENERATED FROM PYTHON SOURCE LINES 178-182

.. code-block:: Python


    print(string_type(cache, with_shape=True))






.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    DynamicCache(key_cache=#2[T1s2x4x3x7,T1s2x4x3x7], value_cache=#2[T1s2x4x3x7,T1s2x4x3x7])




.. GENERATED FROM PYTHON SOURCE LINES 183-198

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








.. GENERATED FROM PYTHON SOURCE LINES 199-200

And the first set of inputs looks like:

.. GENERATED FROM PYTHON SOURCE LINES 200-202

.. code-block:: Python

    print(string_type(inputs[0], with_shape=True))





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    (DynamicCache(key_cache=#2[T1s2x4x3x7,T1s2x4x3x7], value_cache=#2[T1s2x4x3x7,T1s2x4x3x7]),T1s1x1x1x7)




.. GENERATED FROM PYTHON SOURCE LINES 203-204

We can now compute the dynamic shapes.

.. GENERATED FROM PYTHON SOURCE LINES 204-209

.. code-block:: Python


    mi = ModelInputs(Model(), inputs)
    ds = mi.guess_dynamic_shapes()
    pprint.pprint(ds)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    (([[{0: _DimHint(type=<_DimHintType.DYNAMIC: 3>),
         2: _DimHint(type=<_DimHintType.DYNAMIC: 3>),
         3: _DimHint(type=<_DimHintType.DYNAMIC: 3>)},
        {0: _DimHint(type=<_DimHintType.DYNAMIC: 3>),
         2: _DimHint(type=<_DimHintType.DYNAMIC: 3>),
         3: _DimHint(type=<_DimHintType.DYNAMIC: 3>)}],
       [{0: _DimHint(type=<_DimHintType.DYNAMIC: 3>),
         2: _DimHint(type=<_DimHintType.DYNAMIC: 3>),
         3: _DimHint(type=<_DimHintType.DYNAMIC: 3>)},
        {0: _DimHint(type=<_DimHintType.DYNAMIC: 3>),
         2: _DimHint(type=<_DimHintType.DYNAMIC: 3>),
         3: _DimHint(type=<_DimHintType.DYNAMIC: 3>)}]],
      {3: _DimHint(type=<_DimHintType.DYNAMIC: 3>)}),
     {})




.. GENERATED FROM PYTHON SOURCE LINES 210-216

And finally the export.
The export is simple if ``transformers>=4.50``, otherwise,
transformers needs to be patched.
:func:`onnx_diagnostic.torch_export_patches.bypass_export_some_errors`
registers functions to serialize ``DynamicCache``. This one is modified to make
the shape inference implemented in :epkg:`torch` happy.

.. GENERATED FROM PYTHON SOURCE LINES 216-226

.. code-block:: Python


    if has_transformers("4.50"):
        ep = torch.export.export(model, inputs[0], dynamic_shapes=ds[0], strict=False)
    else:
        with bypass_export_some_errors(patch_transformers=True) as modificator:
            ep = torch.export.export(
                model, modificator(inputs[0]), dynamic_shapes=ds[0], strict=False
            )
    print(ep)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    /home/xadupre/vv/this312/lib/python3.12/site-packages/torch/backends/mkldnn/__init__.py:78: UserWarning: TF32 acceleration on top of oneDNN is available for Intel GPUs. The current Torch version does not have Intel GPU Support. (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:148.)
      torch._C._set_onednn_allow_tf32(_allow_tf32)
    ExportedProgram:
        class GraphModule(torch.nn.Module):
            def forward(self, cache_key_cache_0: "f32[s0, 4, s1, s11]", cache_key_cache_1: "f32[s0, 4, s1, s11]", cache_value_cache_0: "f32[s0, 4, s1, s11]", cache_value_cache_1: "f32[s0, 4, s1, s11]", z: "f32[1, 1, 1, s11]"):
                 # File: /home/xadupre/github/onnx-diagnostic/_doc/examples/plot_export_with_dynamic_cache.py:155 in forward, code: z
                add: "f32[s0, 4, s1, s11]" = torch.ops.aten.add.Tensor(z, cache_key_cache_0);  z = cache_key_cache_0 = None
                add_1: "f32[s0, 4, s1, s11]" = torch.ops.aten.add.Tensor(add, cache_key_cache_1);  add = cache_key_cache_1 = None
                add_2: "f32[s0, 4, s1, s11]" = torch.ops.aten.add.Tensor(add_1, cache_value_cache_0);  add_1 = cache_value_cache_0 = None
                add_3: "f32[s0, 4, s1, s11]" = torch.ops.aten.add.Tensor(add_2, cache_value_cache_1);  add_2 = cache_value_cache_1 = None
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
    
    Range constraints: {s0: VR[2, int_oo], s1: VR[2, int_oo], s11: VR[2, int_oo]}





.. GENERATED FROM PYTHON SOURCE LINES 227-229

.. code-block:: Python


    doc.plot_legend("dynamic shapes\nfor cache", "torch.export.export", "tomato")



.. image-sg:: /auto_examples/images/sphx_glr_plot_export_with_dynamic_cache_001.png
   :alt: plot export with dynamic cache
   :srcset: /auto_examples/images/sphx_glr_plot_export_with_dynamic_cache_001.png
   :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 6.234 seconds)


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
