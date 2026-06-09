cmake_minimum_required(VERSION 3.18)

project(franka_control_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/control)

find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
set(FRANKA_SOURCE_ROOT "/home/k324/franka_my_code/libfranka-0.21.2" CACHE PATH "libfranka 0.21.2 source root")
set(FRANKA_BUILD_ROOT "${FRANKA_SOURCE_ROOT}/build-openrobots" CACHE PATH "libfranka 0.21.2 build root")
set(FRANKA_LIBRARY "${FRANKA_BUILD_ROOT}/libfranka.so" CACHE FILEPATH "libfranka 0.21.2 shared library")

pybind11_add_module(_franka_backend src/control/cpp/franka_backend.cpp)
target_include_directories(_franka_backend PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/control/cpp
  ${FRANKA_SOURCE_ROOT}/include
  ${FRANKA_SOURCE_ROOT}/common/include
)
target_link_libraries(_franka_backend PRIVATE ${FRANKA_LIBRARY} Eigen3::Eigen)
target_compile_options(_franka_backend PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(_franka_backend PROPERTIES BUILD_RPATH "${FRANKA_BUILD_ROOT}")
