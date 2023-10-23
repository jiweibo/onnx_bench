include(ExternalProject)

set(GLOG_SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/glog)
set(GLOG_PREFIX_DIR ${CMAKE_BINARY_DIR}/third_party/glog)
set(GLOG_INSTALL_DIR ${CMAKE_BINARY_DIR}/third_party/install/glog)

ExternalProject_Add(
  extern_glog
  PREFIX ${GLOG_PREFIX_DIR}
  SOURCE_DIR ${GLOG_SOURCE_DIR}
  DEPENDS gflags
  CMAKE_ARGS  -DBUILD_SHARED_LIBS=OFF
              -DWITH_GFLAGS=OFF
  CMAKE_CACHE_ARGS
              -DCMAKE_INSTALL_PREFIX:PATH=${GLOG_INSTALL_DIR}
              -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
)

set(GLOG_INCLUDE_DIR ${GLOG_INSTALL_DIR}/include)
set(GLOG_LIB ${GLOG_INSTALL_DIR}/lib/libglog.a)
get_filename_component(GLOG_LIB_DIR ${GLOG_LIB} PATH)

add_library(glog STATIC IMPORTED GLOBAL)
set_property(TARGET glog PROPERTY IMPORTED_LOCATION ${GLOG_LIB})
add_dependencies(glog extern_glog gflags)

include_directories(${GLOG_INCLUDE_DIR})
link_directories(${GLOG_LIB_DIR})