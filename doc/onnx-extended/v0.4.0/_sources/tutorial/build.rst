
Build from source
=================

The packages relies on :epkg:`cmake` to build the C++ extensions.
whether it wrapped with :epkg:`pybind11` or :epkg:`cython`.
Both options are available and can be linked with :epkg:`openmp`,
:epkg:`eigen`, :epkg:`onnxruntime`, :epkg:`CUDA`.
*cmake* is called from `setup.py
<https://github.com/sdpython/onnx-extended/blob/main/setup.py#L198>`_
with two instructions:

* ``python setup.py build_ext --inplace``, the legacy way
* ``pip install -e .``, the new way

By default, *cmake* builds with CUDA if it is available. It can be disabled:

* ``python setup.py build_ext -v --inplace --with-cuda=0``, the legacy way
* ``pip install -e . -v --config-settings="--with-cuda=0"``, the new way (not fully working yet)
* ``pip install -e . -v --global-option "--with-cuda=0"``, the deprecated way
* ``USE_CUDA=0 pip install -e . -v``, the run around way

In case there are multiple versions of CUDA installed, option `cuda-version`
can be specified:

::

    python setup.py build_ext --inplace --cuda-version=12.6

The development versions of :epkg:`onnxruntime` can be used if it was already build
``--ort-version=<version or build path>``. Example:

::

    python setup.py build_ext --inplace --cuda-version=12.6 --ort-version=/home/github/onnxruntime/build/linux_cuda/Release

``--cuda-link=SHARED`` helps reducing the binary size.

.. toctree::
    :maxdepth: 1    
    
    build_cython
    build_pybind11
    build_cuda
    build_ortext
    readings
