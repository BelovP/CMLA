project(yalal)
cmake_minimum_required(VERSION 2.6)

option(BUILD_TESTS      "Build unit tests"              ON)
option(DOUBLE_PRECISION "Doubles instead of floats"     ON)

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
        norms qr lu cholesky)

add_subdirectory(yalal)
add_subdirectory(speed_comparison)
if (BUILD_TESTS)
    add_subdirectory(tests)
endif()