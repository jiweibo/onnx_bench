cmake_minimum_required(VERSION 3.12)

project(cuda_hook CXX)

add_library(cuda SHARED cuda_hook.gen.cc)
target_link_libraries(cuda glog)
set_target_properties(cuda PROPERTIES
    LIBRARY cuda
    SOVERSION 1
)
