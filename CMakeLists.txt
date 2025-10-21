cmake_minimum_required(VERSION 3.18)
project(vs LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MY_BUILD_TESTS ON)
set(BUILD_BENCHMARKS ON)

# optimization flags
add_compile_options("-march=native")
add_compile_options("-ffast-math")
add_compile_options("-O3")
add_compile_options("-funroll-loops")

# add_compile_options("-DDEBUG")

include_directories(${PROJECT_SOURCE_DIR}/include)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(FFTW3 REQUIRED)

# download CPM.cmake (only if missing)
set(CPM_DOWNLOAD_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)
if(NOT EXISTS ${CPM_DOWNLOAD_LOCATION})
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cmake)
  file(
    DOWNLOAD
    https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.40.8/CPM.cmake
    ${CPM_DOWNLOAD_LOCATION}
    EXPECTED_HASH SHA256=78ba32abdf798bc616bab7c73aac32a17bbd7b06ad9e26a6add69de8f3ae4791
  )
endif()
include(${CPM_DOWNLOAD_LOCATION})
CPMAddPackage("gh:doctest/doctest@2.4.9")

set(BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
set(FINUFFT_BUILD_TESTS OFF CACHE BOOL "Whether to build the FINUFFT tests" FORCE)
CPMAddPackage("gh:flatironinstitute/finufft@2.4.1")

install(DIRECTORY include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.hpp")

if(MY_BUILD_TESTS)
  enable_testing()
  add_executable(nudft_test test/nudft.cpp)
  target_link_libraries(nudft_test PUBLIC OpenMP::OpenMP_CXX fftw3 finufft doctest)
  target_include_directories(nudft_test PRIVATE ${doctest_SOURCE_DIR}/doctest)
  add_test(NAME nudft_test COMMAND nudft_test)
endif()

if(BUILD_BENCHMARKS)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    benchmark
    URL https://github.com/google/benchmark/archive/refs/heads/main.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(benchmark)

  add_executable(nudft_vs_nufft benchmark/nudft_vs_nufft.cpp)
  target_link_libraries(nudft_vs_nufft PUBLIC OpenMP::OpenMP_CXX fftw3 finufft benchmark::benchmark)
endif()