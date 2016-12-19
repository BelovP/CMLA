project(yalal)
cmake_minimum_required(VERSION 2.6)

option(BUILD_TESTS      "Build unit tests"              ON)
option(BUILD_BENCHMARKS "Build speed comparison tests"  ON)
option(DOUBLE_PRECISION "Doubles instead of floats"     OFF)

if (DOUBLE_PRECISION)
    add_definitions(-DDOUBLE_PRECISION)
endif()

if (CMAKE_VERSION VERSION_LESS 3.1)
    SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++11)
else()
    set(CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_STANDARD 11)
endif()

set(PACKAGES
        norms qr lu cholesky algorithms)

add_subdirectory(yalal)

if (BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if (BUILD_TESTS)
    add_subdirectory(tests)
endif()