cmake_minimum_required(VERSION 3.20)

project(tcm_fit LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(TCM_ENABLE_OPENMP "Enable OpenMP voxel parallelism when available" ON)

add_executable(
  tcm_fit
  src/mask_utils.cpp
  src/main.cpp
  src/lls.cpp
  src/npy.cpp
  src/post_filter.cpp
  src/preprocess.cpp
  src/rescue_refit.cpp
  src/solver.cpp
  src/tcm_model.cpp
)

target_include_directories(tcm_fit PRIVATE src)

add_executable(
  tcm_fit_tacs
  src/tac_fit_main.cpp
  src/lls.cpp
  src/npy.cpp
  src/solver.cpp
  src/tcm_model.cpp
)

target_include_directories(tcm_fit_tacs PRIVATE src)

if(MSVC)
  target_compile_options(tcm_fit PRIVATE /W4)
  target_compile_options(tcm_fit_tacs PRIVATE /W4)
else()
  target_compile_options(tcm_fit PRIVATE -Wall -Wextra -Wpedantic -O3)
  target_compile_options(tcm_fit_tacs PRIVATE -Wall -Wextra -Wpedantic -O3)
endif()

if(TCM_ENABLE_OPENMP)
  find_package(OpenMP QUIET)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(tcm_fit PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(tcm_fit_tacs PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(tcm_fit PRIVATE TCM_HAS_OPENMP=1)
    target_compile_definitions(tcm_fit_tacs PRIVATE TCM_HAS_OPENMP=1)
  endif()
endif()
