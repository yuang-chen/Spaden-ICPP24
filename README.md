# Bitmap-Based SpMV with Tensor Cores


## How to build
Cmake and CUDA are required. Modification to the CMakeLists.txt files may be necessary, e.g. to change GPU architecture.

Instructions:
1. Download the source code into a folder e.g. Spaden.
3. Inside Spaden and command:

> mkdir build  && cd build && cmake  -DCMAKE_BUILD_TYPE=Release .. && make -j && cd ..


## How to run
For testing Spaden, the executable accepts graphs formatted in `mtx` or binary `csr`. 

### examples

1. Run with input file
> `./build/examples/spmv_float -i /input/path`

> `./build/examples/spmv_half -i /input/path`

