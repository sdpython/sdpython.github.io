
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_examples/plot_export_locate_issue.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_examples_plot_export_locate_issue.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_plot_export_locate_issue.py:


.. _l-plot-export-locale-issue:

==================================================
Find and fix an export issue due to dynamic shapes
==================================================

LLMs must be exported with dynamic shapes and it is common that
a static dimension turns into a static ones. The error message from
:epkg:`pytorch` tells the user to define ``TORCH_LOGS="+dynamic"``
but it shows a very long list of messages where we need
to find the string ``range_refined_to_singleton`` and that
does not really indicates where it comes from. The example
shows how to tweak pytorch to get that information until
it gets better.

A model with an export issue
============================

The following model implies the first dimension of x is equal to 1
or equal to the number of element in the list ``ys``.
It is not really dynamic. It looks obvious here but
it is difficult to find deep inside a big model.

.. GENERATED FROM PYTHON SOURCE LINES 25-44

.. code-block:: Python


    import traceback
    import torch
    from onnx_diagnostic import doc
    from onnx_diagnostic.torch_export_patches import bypass_export_some_errors


    class ModelWithIssue(torch.nn.Module):
        def forward(self, x: torch.Tensor, ys: list[torch.Tensor]):
            caty = torch.cat([y.unsqueeze(0) for y in ys], axis=0)
            z = x * caty
            return z


    inputs = (torch.rand(2, 3, 1), [torch.rand(3, 4), torch.rand(3, 4)])
    model = ModelWithIssue()
    model(*inputs)






.. rst-class:: sphx-glr-script-out

 .. code-block:: none


    tensor([[[0.0516, 0.1260, 0.0155, 0.0400],
             [0.0335, 0.6101, 0.2348, 0.8184],
             [0.7461, 0.4075, 0.4041, 0.2882]],

            [[0.2237, 0.4679, 0.2336, 0.6745],
             [0.5133, 0.2719, 0.3762, 0.3331],
             [0.1130, 0.1312, 0.0498, 0.0553]]])



.. GENERATED FROM PYTHON SOURCE LINES 45-46

Let's export.

.. GENERATED FROM PYTHON SOURCE LINES 46-56

.. code-block:: Python


    DYN = torch.export.Dim.DYNAMIC
    dyn_shapes = ({0: DYN, 1: DYN}, [{0: DYN, 1: DYN}, {0: DYN, 1: DYN}])
    try:
        ep = torch.export.export(model, inputs, dynamic_shapes=dyn_shapes)
        print(ep)
    except Exception as e:
        print("-- ERROR:")
        print(e)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    -- ERROR:
    Constraints violated (L['args'][0][0].size()[0])! For more information, run with TORCH_LOGS="+dynamic".
      - Not all values of RelaxedUnspecConstraint(L['args'][0][0].size()[0]) are valid because L['args'][0][0].size()[0] was inferred to be a constant (2).




.. GENERATED FROM PYTHON SOURCE LINES 57-87

The error shows:

.. code-block::

      Constraints violated (L['args'][0][0].size()[0])!
          For more information, run with TORCH_LOGS="+dynamic".
      - Not all values of RelaxedUnspecConstraint(L['args'][0][0].size()[0])
          are valid because L['args'][0][0].size()[0] was inferred to be a constant (2).

Where does it happens? That's a tricky question we need to answer.
The message is raised from
`torch.fx.experimental.symbolic_shapes.ShapeEnv._set_replacement
<https://github.com/pytorch/pytorch/blob/main/torch/fx/experimental/symbolic_shapes.py#L6239>`_.
One way to find the exact location is to retrieve a stack trace
by inserting an assert such as the following:

.. code-block::

  assert msg != "range_refined_to_singleton", (
      f"A dynamic dimension becomes static! "
      f"a={a!r}, tgt={tgt!r}, msg={msg!r}, tgt_bound={tgt_bound}"
  )

Stop when a dynamic dimension turns static
==========================================

We use :func:`bypass_export_some_errors
<onnx_diagnostic.torch_export_patches.bypass_export_some_errors>`
to replace torch implementation by a new one raising the exception
mentioned in previous section.

.. GENERATED FROM PYTHON SOURCE LINES 87-106

.. code-block:: Python


    with bypass_export_some_errors(stop_if_static=1, verbose=1):
        try:
            torch.export.export(model, inputs, dynamic_shapes=dyn_shapes)
        except (AssertionError, torch._dynamo.exc.TorchRuntimeError) as e:
            print("-- It failed as excepted.")
            print(f"-- final error is {e}")
            print("-- Stack Trace")
            print(traceback.format_exc())

    # The stack trace is quite long but the first line referring to this example
    # is the following one. It points out the line turing a dynamic dimension into
    # static.
    #
    # .. code-block::
    #
    #   File "onnx-diagnostic/_doc/examples/plot_export_locate_issue.py", line 25, in forward
    #       z = x * caty





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    [bypass_export_some_errors] replace torch.jit.isinstance, torch._dynamo.mark_static_address
    [_register_cache_serialization] register <class 'transformers.cache_utils.MambaCache'>
    [_register_cache_serialization] register <class 'transformers.cache_utils.EncoderDecoderCache'>
    [bypass_export_some_errors] sympy.__version__='1.13.3'
    [bypass_export_some_errors] patch sympy
    [bypass_export_some_errors] torch.__version__='2.8.0.dev20250416+cu126'
    [bypass_export_some_errors] stop_if_static=1
    [bypass_export_some_errors] patch pytorch
    [bypass_export_some_errors] modifies shape constraints
    [bypass_export_some_errors] assert when a dynamic dimension turns static
    [bypass_export_some_errors] replaces ShapeEnv._set_replacement
    [bypass_export_some_errors] replaces ShapeEnv._log_guard
    [bypass_export_some_errors] done patching
    -- It failed as excepted.
    -- final error is patched_ShapeEnv: A dynamic dimension becomes static! a=s35, tgt=2, msg='range_refined_to_singleton', tgt_bound=VR[2, 2]
    -- Stack Trace
    Traceback (most recent call last):
      File "/home/xadupre/github/onnx-diagnostic/_doc/examples/plot_export_locate_issue.py", line 90, in <module>
        torch.export.export(model, inputs, dynamic_shapes=dyn_shapes)
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/export/__init__.py", line 275, in export
        return _export(
               ^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/export/_trace.py", line 1109, in wrapper
        raise e
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/export/_trace.py", line 1075, in wrapper
        ep = fn(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/export/exported_program.py", line 122, in wrapper
        return fn(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/export/_trace.py", line 2119, in _export
        ep = _export_for_training(
             ^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/export/_trace.py", line 1109, in wrapper
        raise e
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/export/_trace.py", line 1075, in wrapper
        ep = fn(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/export/exported_program.py", line 122, in wrapper
        return fn(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/export/_trace.py", line 1980, in _export_for_training
        export_artifact = export_func(
                          ^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/export/_trace.py", line 1922, in _non_strict_export
        aten_export_artifact = _to_aten_func(  # type: ignore[operator]
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/export/_trace.py", line 1709, in _export_to_aten_ir_make_fx
        gm, graph_signature = transform(_make_fx_helper)(
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/export/_trace.py", line 1850, in _aot_export_non_strict
        gm, sig = aot_export(wrapped_mod, args, kwargs=kwargs, **flags)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/export/_trace.py", line 1629, in _make_fx_helper
        gm = make_fx(
             ^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 2288, in wrapped
        return make_fx_tracer.trace(f, *args)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 2226, in trace
        return self._trace_inner(f, *args)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 2197, in _trace_inner
        t = dispatch_trace(
            ^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/_compile.py", line 51, in inner
        return disable_fn(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 850, in _fn
        return fn(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 1221, in dispatch_trace
        graph = tracer.trace(root, concrete_args)  # type: ignore[arg-type]
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 1785, in trace
        res = super().trace(root, concrete_args)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 850, in _fn
        return fn(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/fx/_symbolic_trace.py", line 837, in trace
        (self.create_arg(fn(*args)),),
                         ^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 1276, in wrapped
        out = f(*tensors)  # type:ignore[call-arg]
              ^^^^^^^^^^^
      File "<string>", line 1, in <lambda>
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/export/_trace.py", line 1533, in wrapped_fn
        return tuple(flat_fn(*args))
                     ^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/_functorch/_aot_autograd/utils.py", line 184, in flat_fn
        tree_out = fn(*args, **kwargs)
                   ^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/_functorch/_aot_autograd/traced_function_transforms.py", line 903, in functional_call
        out = mod(*args[params_len:], **kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/fx/_symbolic_trace.py", line 812, in module_call_wrapper
        return self.call_module(mod, forward, args, kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 1855, in call_module
        return Tracer.call_module(self, m, forward, args, kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/fx/_symbolic_trace.py", line 530, in call_module
        ret_val = forward(*args, **kwargs)
                  ^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/fx/_symbolic_trace.py", line 805, in forward
        return _orig_module_call(mod, *args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
        return self._call_impl(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
        return forward_call(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/export/_trace.py", line 1834, in forward
        tree_out = mod(*args, **kwargs)
                   ^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/fx/_symbolic_trace.py", line 812, in module_call_wrapper
        return self.call_module(mod, forward, args, kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 1855, in call_module
        return Tracer.call_module(self, m, forward, args, kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/fx/_symbolic_trace.py", line 530, in call_module
        ret_val = forward(*args, **kwargs)
                  ^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/fx/_symbolic_trace.py", line 805, in forward
        return _orig_module_call(mod, *args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
        return self._call_impl(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
        return forward_call(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/github/onnx-diagnostic/_doc/examples/plot_export_locate_issue.py", line 35, in forward
        z = x * caty
            ~~^~~~~~
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 1324, in __torch_function__
        return func(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 1371, in __torch_function__
        return func(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/_export/non_strict_utils.py", line 847, in __torch_function__
        return func(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/_ops.py", line 914, in handler
        return torch._library.utils.handle_dispatch_mode(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/_library/utils.py", line 296, in handle_dispatch_mode
        return curr_mode.__torch_dispatch__(op_overload, overload_types, args, kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/utils/_stats.py", line 27, in wrapper
        return fn(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 1426, in __torch_dispatch__
        return proxy_call(self, func, self.pre_dispatch, args, kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/fx/experimental/proxy_tensor.py", line 926, in proxy_call
        out = func(*args, **kwargs)
              ^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/_ops.py", line 795, in __call__
        return self._op(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/utils/_stats.py", line 27, in wrapper
        return fn(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py", line 1311, in __torch_dispatch__
        return self.dispatch(func, types, args, kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py", line 1932, in dispatch
        return self._cached_dispatch_impl(func, types, args, kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py", line 1423, in _cached_dispatch_impl
        output = self._dispatch_impl(func, types, args, kwargs)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/_subclasses/fake_tensor.py", line 2440, in _dispatch_impl
        return maybe_propagate_real_tensors(fast_impl(self, *args, **kwargs))
                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/_subclasses/fake_impls.py", line 957, in fast_binary_impl
        final_shape = infer_size(final_shape, shape)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/github/onnx-diagnostic/onnx_diagnostic/torch_export_patches/patches/patch_torch.py", line 117, in patched_infer_size
        b3 = guard_size_oblivious(sizeA == sizeB)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/fx/experimental/symbolic_shapes.py", line 414, in guard_size_oblivious
        return expr.node.guard_size_oblivious("", 0)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/fx/experimental/sym_node.py", line 588, in guard_size_oblivious
        r = self.evaluate(size_oblivious=True)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/fx/experimental/sym_node.py", line 510, in evaluate
        return self.shape_env.evaluate_sym_node(self, size_oblivious)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/fx/experimental/symbolic_shapes.py", line 6776, in evaluate_sym_node
        return self.evaluate_expr(
               ^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/fx/experimental/recording.py", line 263, in wrapper
        return retlog(fn(*args, **kwargs))
                      ^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/fx/experimental/symbolic_shapes.py", line 6792, in evaluate_expr
        return self._evaluate_expr(
               ^^^^^^^^^^^^^^^^^^^^
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/fx/experimental/symbolic_shapes.py", line 7107, in _evaluate_expr
        self._maybe_guard_rel(g)
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/fx/experimental/symbolic_shapes.py", line 6426, in _maybe_guard_rel
        self._refine_ranges(expr)
      File "/home/xadupre/vv/this312/lib/python3.12/site-packages/torch/fx/experimental/symbolic_shapes.py", line 7311, in _refine_ranges
        self._set_replacement(
      File "/home/xadupre/github/onnx-diagnostic/onnx_diagnostic/torch_export_patches/patches/patch_torch.py", line 341, in _set_replacement
        assert msg != "range_refined_to_singleton", (
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: patched_ShapeEnv: A dynamic dimension becomes static! a=s35, tgt=2, msg='range_refined_to_singleton', tgt_bound=VR[2, 2]

    [bypass_export_some_errors] remove patches
    [bypass_export_some_errors] restored sympy functions
    [bypass_export_some_errors] restored pytorch functions
    [bypass_export_some_errors] restored ShapeEnv._set_replacement
    [bypass_export_some_errors] restored ShapeEnv._log_guard
    [bypass_export_some_errors] restored shape constraints
    [_unregister_cache_serialization] unregistered MambaCache
    [_unregister_cache_serialization] unregistered EncoderDecoderCache




.. GENERATED FROM PYTHON SOURCE LINES 107-111

.. code-block:: Python


    doc.plot_legend(
        "dynamic dimension\nwas inferred\nto be a constant", "torch.export.export", "tomato"
    )



.. image-sg:: /auto_examples/images/sphx_glr_plot_export_locate_issue_001.png
   :alt: plot export locate issue
   :srcset: /auto_examples/images/sphx_glr_plot_export_locate_issue_001.png
   :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 0.196 seconds)


.. _sphx_glr_download_auto_examples_plot_export_locate_issue.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_export_locate_issue.ipynb <plot_export_locate_issue.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_export_locate_issue.py <plot_export_locate_issue.py>`

    .. container:: sphx-glr-download sphx-glr-download-zip

      :download:`Download zipped: plot_export_locate_issue.zip <plot_export_locate_issue.zip>`


.. include:: plot_export_locate_issue.recommendations


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
