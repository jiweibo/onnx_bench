include(ExternalProject)

set(CNPY_SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/cnpy)
set(CNPY_PREFIX_DIR ${CMAKE_BINARY_DIR}/third_party/cnpy)
set(CNPY_INSTALL_DIR ${CMAKE_BINARY_DIR}/third_party/install/cnpy)

ExternalProject_Add(
  extern_cnpy
  PREFIX ${CNPY_PREFIX_DIR}
  SOURCE_DIR ${CNPY_SOURCE_DIR}
  DEPENDS gflags
  CMAKE_ARGS  -DBUILD_SHARED_LIBS=OFF
  CMAKE_CACHE_ARGS
              -DCMAKE_INSTALL_PREFIX:PATH=${CNPY_INSTALL_DIR}
              -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
)

set(CNPY_INCLUDE_DIR ${CNPY_INSTALL_DIR}/include)
set(CNPY_LIB ${CNPY_INSTALL_DIR}/lib/libcnpy.a)
get_filename_component(CNPY_LIB_DIR ${CNPY_LIB} PATH)

add_library(cnpy STATIC IMPORTED GLOBAL)
set_property(TARGET cnpy PROPERTY IMPORTED_LOCATION ${CNPY_LIB})
add_dependencies(cnpy extern_cnpy)

include_directories(${CNPY_INCLUDE_DIR})
link_directories(${CNPY_LIB_DIR})