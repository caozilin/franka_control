cmake_minimum_required(VERSION 3.18)

project(franka_control_backend LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/control)

if(DEFINED PYTHON_EXECUTABLE AND NOT DEFINED Python_EXECUTABLE)
  set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
endif()
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
  RESULT_VARIABLE PYBIND11_CMAKEDIR_RESULT
  OUTPUT_VARIABLE PYBIND11_CMAKEDIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(PYBIND11_CMAKEDIR_RESULT EQUAL 0)
  list(PREPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKEDIR}")
endif()

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
get_filename_component(
  DEFAULT_FRANKA_SOURCE_ROOT
  "${CMAKE_CURRENT_SOURCE_DIR}/../libfranka"
  ABSOLUTE
)
set(FRANKA_SOURCE_ROOT "${DEFAULT_FRANKA_SOURCE_ROOT}" CACHE PATH "libfranka 0.21.2 source root")
set(FRANKA_BUILD_ROOT "${FRANKA_SOURCE_ROOT}/build-openrobots" CACHE PATH "libfranka 0.21.2 build root")
set(FRANKA_LIBRARY "${FRANKA_BUILD_ROOT}/libfranka.so" CACHE FILEPATH "libfranka 0.21.2 shared library")

if(NOT EXISTS "${FRANKA_SOURCE_ROOT}/include/franka/robot.h")
  message(FATAL_ERROR
    "libfranka headers were not found at ${FRANKA_SOURCE_ROOT}. "
    "Place the libfranka folder next to this project, at ${DEFAULT_FRANKA_SOURCE_ROOT}."
  )
endif()

if(NOT EXISTS "${FRANKA_LIBRARY}")
  message(FATAL_ERROR
    "libfranka was not built at ${FRANKA_LIBRARY}. "
    "Configure and build ${FRANKA_SOURCE_ROOT} first."
  )
endif()

pybind11_add_module(_franka_backend src/control/cpp/franka_backend.cpp)
target_include_directories(_franka_backend PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/control/cpp
  ${FRANKA_SOURCE_ROOT}/include
  ${FRANKA_SOURCE_ROOT}/common/include
)
target_link_libraries(_franka_backend PRIVATE ${FRANKA_LIBRARY} Eigen3::Eigen)
target_compile_options(_franka_backend PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(_franka_backend PROPERTIES BUILD_RPATH "${FRANKA_BUILD_ROOT}")
