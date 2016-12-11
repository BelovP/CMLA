find_package(OpenCV 3 REQUIRED COMPONENTS core)
include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCES "")
foreach (PACKAGE ${PACKAGES})
    list(APPEND SOURCES ${PACKAGE}.cpp)
endforeach()

add_library(yalal SHARED ${SOURCES})
target_link_libraries(yalal ${OpenCV_LIBS})