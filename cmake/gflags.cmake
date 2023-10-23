include(ExternalProject)

set(GFLAGS_SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/gflags)
set(GFLAGS_PREFIX_DIR ${CMAKE_BINARY_DIR}/third_party/gflags)
set(GFLAGS_INSTALL_DIR ${CMAKE_BINARY_DIR}/third_party/install/gflags)

ExternalProject_Add(
  extern_gflags
  PREFIX ${GFLAGS_PREFIX_DIR}
  SOURCE_DIR ${GFLAGS_SOURCE_DIR}
  CMAKE_ARGS  -DBUILD_SHARED_LIBS=OFF
              -DBUILD_STATIC_LIBS=ON
  CMAKE_CACHE_ARGS
              -DCMAKE_INSTALL_PREFIX:PATH=${GFLAGS_INSTALL_DIR}
              -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
)

set(GFLAGS_INCLUDE_DIR ${GFLAGS_INSTALL_DIR}/include)
set(GFLAGS_LIB ${GFLAGS_INSTALL_DIR}/lib/libgflags.a)
get_filename_component(GFLAGS_LIB_DIR ${GFLAGS_LIB} PATH)

add_library(gflags STATIC IMPORTED GLOBAL)
set_property(TARGET gflags PROPERTY IMPORTED_LOCATION ${GFLAGS_LIB})
add_dependencies(gflags extern_gflags)

include_directories(${GFLAGS_INCLUDE_DIR})
link_directories(${GFLAGS_LIB_DIR})