include(ExternalProject)

set(JSONCPP_SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/jsoncpp)
set(JSONCPP_PREFIX_DIR ${CMAKE_BINARY_DIR}/third_party/jsoncpp)
set(JSONCPP_INSTALL_DIR ${CMAKE_BINARY_DIR}/third_party/install/jsoncpp)

ExternalProject_Add(
  extern_jsoncpp
  PREFIX ${JSONCPP_PREFIX_DIR}
  SOURCE_DIR ${JSONCPP_SOURCE_DIR}
  CMAKE_ARGS  -DJSONCPP_WITH_TESTS=OFF
              -DBUILD_SHARED_LIBS=OFF
              -DBUILD_STATIC_LIBS=ON
  CMAKE_CACHE_ARGS
              -DCMAKE_INSTALL_PREFIX:PATH=${JSONCPP_INSTALL_DIR}
              -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
)

set(JSONCPP_INCLUDE_DIR ${JSONCPP_INSTALL_DIR}/include)
set(JSONCPP_LIB ${JSONCPP_INSTALL_DIR}/lib/libjsoncpp.a)
get_filename_component(JSONCPP_LIB_DIR ${JSONCPP_LIB} PATH)

add_library(jsoncpp STATIC IMPORTED GLOBAL)
set_property(TARGET jsoncpp PROPERTY IMPORTED_LOCATION ${JSONCPP_LIB})
add_dependencies(jsoncpp extern_jsoncpp)

include_directories(${JSONCPP_INCLUDE_DIR})
link_directories(${JSONCPP_LIB_DIR})