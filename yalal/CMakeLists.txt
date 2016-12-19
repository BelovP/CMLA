find_package(OpenCV 3 REQUIRED COMPONENTS core)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(OpenMP)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS})

set(SOURCES "")
foreach (PACKAGE ${PACKAGES})
    list(APPEND SOURCES ${PACKAGE}.cpp)
endforeach()

add_library(yalal SHARED ${SOURCES})
target_link_libraries(yalal ${OpenCV_LIBS})