add_library(bmplib INTERFACE)
target_include_directories(bmplib INTERFACE .)

target_link_libraries(bmplib INTERFACE
  OpenMP::OpenMP_CXX
  CUDA::cusparse
)

target_compile_options(bmplib INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:
  -Xcompiler=-fopenmp # OpenMP flags for CUDA compiler
  --extended-lambda
  --expt-relaxed-constexpr
  --generate-line-info
  >
)