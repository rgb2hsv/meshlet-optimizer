cmake_minimum_required(VERSION 3.21)
project(meshlet_optimizer LANGUAGES C CXX)

option(MOPT_OPENMP "Parallelize with OpenMP" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # Compile the sources of each target in parallel (Visual Studio generators; Ninja already runs in parallel).
  add_compile_options(/MP)
endif()

set(EXT ${CMAKE_CURRENT_SOURCE_DIR}/external)
foreach(dep eigen cereal fmt happly argparse CDT LBFGSpp meshoptimizer kdtree-cpp)
  if(NOT EXISTS ${EXT}/${dep}/.git)
    message(FATAL_ERROR "external/${dep} is missing: run git submodule update --init --recursive")
  endif()
endforeach()

# Compiled dependencies: fmt and meshoptimizer (static, their own tests and tools off).
set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(${EXT}/fmt EXCLUDE_FROM_ALL)
add_subdirectory(${EXT}/meshoptimizer EXCLUDE_FROM_ALL)

# Header-only dependencies.
add_library(mopt_deps INTERFACE)
target_include_directories(mopt_deps SYSTEM INTERFACE
  ${EXT}/eigen
  ${EXT}/cereal/include
  ${EXT}/happly
  ${EXT}/argparse/include
  ${EXT}/CDT/CDT/include
  ${EXT}/LBFGSpp/include
  ${EXT}/kdtree-cpp/src
  ${EXT}/meshoptimizer)  # demo/clusterlod.h
target_link_libraries(mopt_deps INTERFACE fmt::fmt meshoptimizer)

set(MOPT_MODULES core mesh charts flattening hierarchy mlod)
set(core_sources)
set(core_headers)
foreach(module ${MOPT_MODULES})
  file(GLOB module_sources CONFIGURE_DEPENDS src/${module}/*.cpp)
  file(GLOB module_headers CONFIGURE_DEPENDS src/${module}/*.hpp)
  list(APPEND core_sources ${module_sources})
  list(APPEND core_headers ${module_headers})
endforeach()
add_library(mopt_core STATIC ${core_sources} ${core_headers})
target_include_directories(mopt_core PUBLIC src)
target_link_libraries(mopt_core PUBLIC mopt_deps)
target_compile_definitions(mopt_core PUBLIC
  EIGEN_INITIALIZE_MATRICES_BY_ZERO
  $<$<CXX_COMPILER_ID:MSVC>:_USE_MATH_DEFINES NOMINMAX _CRT_SECURE_NO_WARNINGS>)

if(MSVC)
  target_compile_options(mopt_core PUBLIC /bigobj /utf-8 /Zc:__cplusplus /W3 /wd4267 /wd4244 /wd4305 /wd4838 /wd4996)
else()
  target_compile_options(mopt_core PUBLIC -Wall -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable
                                          -Wno-unused-function -Wno-deprecated-copy)
endif()

if(MOPT_OPENMP)
  if(APPLE AND NOT OpenMP_ROOT)
    # Apple clang has no OpenMP runtime of its own: use Homebrew's libomp when it is there.
    foreach(prefix /opt/homebrew/opt/libomp /usr/local/opt/libomp)
      if(EXISTS ${prefix}/include/omp.h)
        set(OpenMP_ROOT ${prefix})
        break()
      endif()
    endforeach()
  endif()
  if(MSVC)
    # The default /openmp is OpenMP 2.0 (signed loop indices only); LLVM's runtime supports 3.x+.
    set(OpenMP_RUNTIME_MSVC llvm)
  endif()
  find_package(OpenMP COMPONENTS CXX)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(mopt_core PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP ${OpenMP_CXX_VERSION}: enabled")
  else()
    message(WARNING "OpenMP not found: building single-threaded (macOS: brew install libomp)")
  endif()
endif()

add_executable(meshlet-optimizer src/app/main.cpp src/app/Options.cpp src/app/Options.hpp)
target_link_libraries(meshlet-optimizer PRIVATE mopt_core)
set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT meshlet-optimizer)

enable_testing()
add_test(NAME metasphere
         COMMAND meshlet-optimizer ${CMAKE_CURRENT_SOURCE_DIR}/assets/model/metasphere.ply
                 -o ${CMAKE_CURRENT_BINARY_DIR}/test/metasphere.mlod --check)
