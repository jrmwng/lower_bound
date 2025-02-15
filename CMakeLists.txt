cmake_minimum_required(VERSION 3.10)

project(lower_bound_project)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(include)

add_executable(lower_bound_test src/main.cpp)
add_executable(lower_bound_tests tests/test_lower_bound.cpp)
add_executable(lower_bound_tests_simd tests/test_lower_bound_simd.cpp)

# Add Google Test
add_subdirectory(googletest)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

target_link_libraries(lower_bound_tests gtest gtest_main)
target_link_libraries(lower_bound_tests_simd gtest gtest_main)

# Add AVX2 support
if (MSVC)
    target_compile_options(lower_bound_test PRIVATE /arch:AVX2)
    target_compile_options(lower_bound_tests PRIVATE /arch:AVX2)
    target_compile_options(lower_bound_tests_simd PRIVATE /arch:AVX2)
else()
    target_compile_options(lower_bound_test PRIVATE -mavx2)
    target_compile_options(lower_bound_tests PRIVATE -mavx2)
    target_compile_options(lower_bound_tests_simd PRIVATE -mavx2)
endif()

enable_testing()
add_test(NAME LowerBoundTests COMMAND lower_bound_tests)
add_test(NAME LowerBoundTestsSimd COMMAND lower_bound_tests_simd)