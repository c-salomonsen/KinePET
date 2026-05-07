cmake_minimum_required(VERSION 3.20)

project(tcm_fit LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

option(TCM_ENABLE_OPENMP "Enable OpenMP TAC parallelism when available" ON)
option(TCM_ENABLE_PYTHON_BINDINGS "Build the nanobind Python extension" ON)

add_library(
  tcm_solver_core
  src/lls.cpp
  src/solver.cpp
  src/tac_batch.cpp
  src/tcm_model.cpp
)

target_include_directories(tcm_solver_core PUBLIC src)

if(MSVC)
  target_compile_options(tcm_solver_core PRIVATE /W4 /O2)
else()
  target_compile_options(tcm_solver_core PRIVATE -Wall -Wextra -Wpedantic -O3)
endif()

if(TCM_ENABLE_OPENMP)
  find_package(OpenMP QUIET)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(tcm_solver_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(tcm_solver_core PUBLIC TCM_HAS_OPENMP=1)
  endif()
endif()

add_executable(
  tcm_fit_tacs
  src/npy.cpp
  src/tac_fit_main.cpp
)

target_link_libraries(tcm_fit_tacs PRIVATE tcm_solver_core)
target_include_directories(tcm_fit_tacs PRIVATE src)

if(MSVC)
  target_compile_options(tcm_fit_tacs PRIVATE /W4 /O2)
else()
  target_compile_options(tcm_fit_tacs PRIVATE -Wall -Wextra -Wpedantic -O3)
endif()

if(TCM_ENABLE_PYTHON_BINDINGS)
  if (CMAKE_VERSION VERSION_LESS 3.18)
    set(DEV_MODULE Development)
  else()
    set(DEV_MODULE Development.Module)
  endif()

  find_package(Python 3.10 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    RESULT_VARIABLE TCM_NANOBIND_STATUS
    OUTPUT_VARIABLE nanobind_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if (NOT TCM_NANOBIND_STATUS EQUAL 0)
    message(FATAL_ERROR "Could not locate nanobind. Install it in the active Python environment first.")
  endif()

  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(
    _core
    NB_STATIC
    src/python_bindings.cpp
  )

  target_link_libraries(_core PRIVATE tcm_solver_core)
  target_include_directories(_core PRIVATE src)

  install(TARGETS _core LIBRARY DESTINATION kinepet)
endif()
