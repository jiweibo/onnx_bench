add_executable(test_core test_core.cc gtest_main.cc)
target_link_libraries(test_core ${DEPS} gtest pthread)
add_test(NAME test_core COMMAND test_core)

add_executable(test_trt_net test_trt_net.cc)
target_link_libraries(test_trt_net trt_lib ${DEPS})

add_executable(test_trt test_trt.cc)
target_link_libraries(test_trt ${DEPS} nvinfer nvinfer_plugin nvonnxparser)

add_executable(submit_multi_kernel submit_multi_kernel.cu)
target_link_libraries(submit_multi_kernel ${DEPS})
