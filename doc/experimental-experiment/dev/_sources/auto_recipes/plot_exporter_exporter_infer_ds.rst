
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_recipes/plot_exporter_exporter_infer_ds.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_recipes_plot_exporter_exporter_infer_ds.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_recipes_plot_exporter_exporter_infer_ds.py:


.. _l-plot-exporter-exporter-infer-ds:

Infer dynamic shapes before exporting
=====================================

Dynamic shapes need to be specified to get a model able to cope
with different dimensions. Input rank are expected to be the same
but the dimension may change. The user has the ability to
set them up or to call a function able to infer them from
two sets of inputs having different values for the dynamic dimensions.

Infer dynamic shapes
++++++++++++++++++++

.. GENERATED FROM PYTHON SOURCE LINES 16-56

.. code-block:: Python


    import onnx
    from onnx_array_api.plotting.graphviz_helper import plot_dot
    import torch
    from experimental_experiment.torch_interpreter import to_onnx
    from experimental_experiment.torch_interpreter.piece_by_piece import (
        trace_execution_piece_by_piece,
    )


    class MA(torch.nn.Module):
        def forward(self, x, y):
            return x + y


    class MM(torch.nn.Module):
        def forward(self, x, y):
            return x * y


    class MASMM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ma = MA()
            self.mm = MM()

        def forward(self, x, y, z):
            return self.ma(x, y) - self.mm(y, z)


    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ma = MA()
            self.masmm = MASMM()

        def forward(self, x):
            return self.ma(x, self.masmm(x, x, x))









.. GENERATED FROM PYTHON SOURCE LINES 57-58

The model.

.. GENERATED FROM PYTHON SOURCE LINES 58-60

.. code-block:: Python

    model = Model()








.. GENERATED FROM PYTHON SOURCE LINES 61-62

Two sets of inputs.

.. GENERATED FROM PYTHON SOURCE LINES 62-67

.. code-block:: Python

    inputs = [
        ((torch.randn((5, 6)),), {}),
        ((torch.randn((6, 6)),), {}),
    ]








.. GENERATED FROM PYTHON SOURCE LINES 68-70

Then we run the model, stores intermediates inputs and outputs,
to finally guess the dynamic shapes.

.. GENERATED FROM PYTHON SOURCE LINES 70-74

.. code-block:: Python

    diag = trace_execution_piece_by_piece(model, inputs, verbose=0)
    pretty = diag.pretty_text(with_dynamic_shape=True)
    print(pretty)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    >>> __main__: Model
      DS=(({0: _DimHint(type=<_DimHintType.DYNAMIC: 3>, min=None, max=None, _factory=True)},), {})
      > ((CT1s5x6[-1.910679578781128,2.5698020458221436:A0.05161839084078868],),{})
      > ((CT1s6x6[-1.8676968812942505,1.7400524616241455:A0.08039881030304565],),{})
        >>> ma: MA
          DS=(({0: _DimHint(type=<_DimHintType.DYNAMIC: 3>, min=None, max=None, _factory=True)}, {0: _DimHint(type=<_DimHintType.DYNAMIC: 3>, min=None, max=None, _factory=True)}), {})
          > ((CT1s5x6[-1.910679578781128,2.5698020458221436:A0.05161839084078868],CT1s5x6[-7.472055435180664,0.999890148639679:A-0.7152137871831655]),{})
          > ((CT1s6x6[-1.8676968812942505,1.7400524616241455:A0.08039881030304565],CT1s6x6[-7.223685264587402,0.9954756498336792:A-0.6076143090095785]),{})
          < (CT1s5x6[-9.382735252380371,2.167281150817871:A-0.6635954144100348],)
          < (CT1s6x6[-9.091382026672363,2.22957706451416:A-0.5272154894967874],)
        <<<
        >>> masmm: MASMM
          DS=(({0: _DimHint(type=<_DimHintType.DYNAMIC: 3>, min=None, max=None, _factory=True)}, {0: _DimHint(type=<_DimHintType.DYNAMIC: 3>, min=None, max=None, _factory=True)}, {0: _DimHint(type=<_DimHintType.DYNAMIC: 3>, min=None, max=None, _factory=True)}), {})
          > ((CT1s5x6[-1.910679578781128,2.5698020458221436:A0.05161839084078868],CT1s5x6[-1.910679578781128,2.5698020458221436:A0.05161839084078868],CT1s5x6[-1.910679578781128,2.5698020458221436:A0.05161839084078868]),{})
          > ((CT1s6x6[-1.8676968812942505,1.7400524616241455:A0.08039881030304565],CT1s6x6[-1.8676968812942505,1.7400524616241455:A0.08039881030304565],CT1s6x6[-1.8676968812942505,1.7400524616241455:A0.08039881030304565]),{})
            >>> ma: MA
              DS=(({0: _DimHint(type=<_DimHintType.DYNAMIC: 3>, min=None, max=None, _factory=True)}, {0: _DimHint(type=<_DimHintType.DYNAMIC: 3>, min=None, max=None, _factory=True)}), {})
              > ((CT1s5x6[-1.910679578781128,2.5698020458221436:A0.05161839084078868],CT1s5x6[-1.910679578781128,2.5698020458221436:A0.05161839084078868]),{})
              > ((CT1s6x6[-1.8676968812942505,1.7400524616241455:A0.08039881030304565],CT1s6x6[-1.8676968812942505,1.7400524616241455:A0.08039881030304565]),{})
              < (CT1s5x6[-3.821359157562256,5.139604091644287:A0.10323678168157736],)
              < (CT1s6x6[-3.735393762588501,3.480104923248291:A0.1607976206060913],)
            <<<
            >>> mm: MM
              DS=(({0: _DimHint(type=<_DimHintType.DYNAMIC: 3>, min=None, max=None, _factory=True)}, {0: _DimHint(type=<_DimHintType.DYNAMIC: 3>, min=None, max=None, _factory=True)}), {})
              > ((CT1s5x6[-1.910679578781128,2.5698020458221436:A0.05161839084078868],CT1s5x6[-1.910679578781128,2.5698020458221436:A0.05161839084078868]),{})
              > ((CT1s6x6[-1.8676968812942505,1.7400524616241455:A0.08039881030304565],CT1s6x6[-1.8676968812942505,1.7400524616241455:A0.08039881030304565]),{})
              < (CT1s5x6[0.0002720642078202218,6.603882789611816:A0.8184505777423813],)
              < (CT1s6x6[0.0008787743281573057,3.4882917404174805:A0.7684119339295042],)
            <<<
          < (CT1s5x6[-7.472055435180664,0.999890148639679:A-0.7152137871831655],)
          < (CT1s6x6[-7.223685264587402,0.9954756498336792:A-0.6076143090095785],)
        <<<
      < (CT1s5x6[-9.382735252380371,2.167281150817871:A-0.6635954144100348],)
      < (CT1s6x6[-9.091382026672363,2.22957706451416:A-0.5272154894967874],)
    <<<




.. GENERATED FROM PYTHON SOURCE LINES 75-76

The dynamic shapes are obtained with:

.. GENERATED FROM PYTHON SOURCE LINES 76-79

.. code-block:: Python

    ds = diag.guess_dynamic_shapes()
    print(ds)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    (({0: _DimHint(type=<_DimHintType.DYNAMIC: 3>, min=None, max=None, _factory=True)},), {})




.. GENERATED FROM PYTHON SOURCE LINES 80-84

Export
++++++

We use these dynamic shapes to export.

.. GENERATED FROM PYTHON SOURCE LINES 84-88

.. code-block:: Python


    ep = torch.export.export(model, inputs[0][0], kwargs=inputs[0][1], dynamic_shapes=ds[0])
    print(ep)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    ExportedProgram:
        class GraphModule(torch.nn.Module):
            def forward(self, x: "f32[s35, 6]"):
                 # File: /home/xadupre/github/experimental-experiment/_doc/recipes/plot_exporter_exporter_infer_ds.py:28 in forward, code: return x + y
                add: "f32[s35, 6]" = torch.ops.aten.add.Tensor(x, x)
            
                 # File: /home/xadupre/github/experimental-experiment/_doc/recipes/plot_exporter_exporter_infer_ds.py:33 in forward, code: return x * y
                mul: "f32[s35, 6]" = torch.ops.aten.mul.Tensor(x, x)
            
                 # File: /home/xadupre/github/experimental-experiment/_doc/recipes/plot_exporter_exporter_infer_ds.py:43 in forward, code: return self.ma(x, y) - self.mm(y, z)
                sub: "f32[s35, 6]" = torch.ops.aten.sub.Tensor(add, mul);  add = mul = None
            
                 # File: /home/xadupre/github/experimental-experiment/_doc/recipes/plot_exporter_exporter_infer_ds.py:28 in forward, code: return x + y
                add_1: "f32[s35, 6]" = torch.ops.aten.add.Tensor(x, sub);  x = sub = None
                return (add_1,)
            
    Graph signature: 
        # inputs
        x: USER_INPUT
    
        # outputs
        add_1: USER_OUTPUT
    
    Range constraints: {s35: VR[2, int_oo]}





.. GENERATED FROM PYTHON SOURCE LINES 89-90

We can use that graph to get the onnx model.

.. GENERATED FROM PYTHON SOURCE LINES 90-95

.. code-block:: Python


    onx, builder = to_onnx(ep, return_builder=True)
    onnx.save(onx, "plot_exporter_exporter_infer_ds.onnx")
    print(builder.pretty_text())





.. rst-class:: sphx-glr-script-out

 .. code-block:: none


    dyn---: s35 -> 's35'
    opset: : 18
    input:: x                                                                       |T1: s35 x 6
    Add: x, x -> add                                                                |T1: s35 x 6                  - add_Tensor
    Mul: x, x -> mul                                                                |T1: s35 x 6                  - mul_Tensor
    Sub: add, mul -> sub                                                            |T1: s35 x 6                  - sub_Tensor
    Add: x, sub -> output_0                                                         |T1: s35 x 6                  - add_Tensor2
    output:: output_0                                                               |T1: s35 x 6




.. GENERATED FROM PYTHON SOURCE LINES 96-97

And visually.

.. GENERATED FROM PYTHON SOURCE LINES 97-99

.. code-block:: Python


    plot_dot(onx)



.. image-sg:: /auto_recipes/images/sphx_glr_plot_exporter_exporter_infer_ds_001.png
   :alt: plot exporter exporter infer ds
   :srcset: /auto_recipes/images/sphx_glr_plot_exporter_exporter_infer_ds_001.png
   :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 2.557 seconds)


.. _sphx_glr_download_auto_recipes_plot_exporter_exporter_infer_ds.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_exporter_exporter_infer_ds.ipynb <plot_exporter_exporter_infer_ds.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_exporter_exporter_infer_ds.py <plot_exporter_exporter_infer_ds.py>`

    .. container:: sphx-glr-download sphx-glr-download-zip

      :download:`Download zipped: plot_exporter_exporter_infer_ds.zip <plot_exporter_exporter_infer_ds.zip>`


.. include:: plot_exporter_exporter_infer_ds.recommendations


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
