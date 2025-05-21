=======================
Coverage of the Patches
=======================

Serialized Classes
==================

The following code shows the list of serialized classes in transformers.

.. runpython::
    :showcode:

    import onnx_diagnostic.torch_export_patches.onnx_export_serialization as p

    print('\n'.join(sorted(p.serialization_functions())))

Patched Classes
===============

The following script shows the list of method patched
for transformers.

.. runpython::
    :showcode:

    import onnx_diagnostic.torch_export_patches.patches.patch_transformers as p

    for name, cls in p.__dict__.items():
        if name.startswith("patched_"):
            print(f"{cls._PATCHED_CLASS_.__name__}: {', '.join(cls._PATCHES_)}")

Half Automated Rewrites
=======================

The following script shows the list of method patched
for transformers.

.. runpython::
    :showcode:

    import onnx_diagnostic.torch_export_patches.patch_module_helper as p

    for name, f in p.__dict__.items():
        if name.startswith("_rewrite_"):
            print(f.__doc__)
