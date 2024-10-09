include(ExternalProject)

set(GTEST_SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/googletest)
set(GTEST_PREFIX_DIR ${CMAKE_BINARY_DIR}/third_party/googletest)
set(GTEST_INSTALL_DIR ${CMAKE_BINARY_DIR}/third_party/install/googletest)

ExternalProject_Add(
  extern_gtest
  PREFIX ${GTEST_PREFIX_DIR}
  SOURCE_DIR ${GTEST_SOURCE_DIR}
  CMAKE_ARGS  -DBUILD_SHARED_LIBS=OFF
              -DBUILD_GMOCK=OFF
  CMAKE_CACHE_ARGS
              -DCMAKE_INSTALL_PREFIX:PATH=${GTEST_INSTALL_DIR}
              -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
)

set(GTEST_INCLUDE_DIR ${GTEST_INSTALL_DIR}/include)
set(GTEST_LIB ${GTEST_INSTALL_DIR}/lib/libgtest.a)
get_filename_component(GTEST_LIB_DIR ${GTEST_LIB} PATH)

add_library(gtest STATIC IMPORTED GLOBAL)
set_property(TARGET gtest PROPERTY IMPORTED_LOCATION ${GTEST_LIB})
add_dependencies(gtest extern_gtest)

include_directories(${GTEST_INCLUDE_DIR})
link_directories(${GTEST_LIB_DIR})