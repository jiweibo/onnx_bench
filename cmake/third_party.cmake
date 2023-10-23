include(cmake/gflags.cmake)
include(cmake/glog.cmake)
include(cmake/jsoncpp.cmake)
include(cmake/cnpy.cmake)

add_custom_target(third_party ALL)
add_dependencies(third_party gflags glog jsoncpp cnpy)