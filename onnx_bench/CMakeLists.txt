add_executable(onnx_bench onnx_bench.cc)
target_link_libraries(onnx_bench onnxruntime)
target_link_libraries(onnx_bench ${DEPS})

# add_executable(demo simple_demo.cc)
# target_link_libraries(demo onnxruntime)
# target_link_libraries(demo ${DEPS})
