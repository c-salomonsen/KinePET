cmake_minimum_required(VERSION 3.20)

project(tcm_fit LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(TCM_ENABLE_OPENMP "Enable OpenMP voxel parallelism when available" ON)

add_executable(
  tcm_fit
  src/main.cpp
  src/lls.cpp
  src/npy.cpp
  src/solver.cpp
  src/tcm_model.cpp
)

target_include_directories(tcm_fit PRIVATE src)

if(MSVC)
  target_compile_options(tcm_fit PRIVATE /W4)
else()
  target_compile_options(tcm_fit PRIVATE -Wall -Wextra -Wpedantic -O3)
endif()

if(TCM_ENABLE_OPENMP)
  find_package(OpenMP QUIET)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(tcm_fit PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(tcm_fit PRIVATE TCM_HAS_OPENMP=1)
  endif()
endif()
