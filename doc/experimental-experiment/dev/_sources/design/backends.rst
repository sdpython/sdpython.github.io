===============
Dynamo Backends
===============
 
The main backend relies on :epkg:`onnxruntime` to execute the ONNX graphs
through function
:func:`onnx_custom_backend <experimental_experiment.torch_dynamo.onnx_custom_backend>`
can be customized to specify different parameter when launching the model.

This function can be replaced by 
:func:`onnx_debug_backend <experimental_experiment.torch_dynamo.onnx_debug_backend>`
to use the onnx reference implementation based on numpy and show the intermediate
results.
