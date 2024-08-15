
add_executable(spmv_half spmv_half.cu)
add_executable(spmv_float spmv_float.cu)

target_link_libraries(spmv_float PRIVATE bmplib)
target_link_libraries(spmv_half PRIVATE bmplib)