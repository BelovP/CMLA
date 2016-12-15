include_directories(${PROJECT_SOURCE_DIR}/yalal)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/extern/Eigen)

add_executable(time_qr time_qr.cpp)
target_link_libraries(time_qr yalal ${ARMADILLO_LIBRARIES})

add_executable(time_lu time_lu.cpp)
target_link_libraries(time_lu yalal ${ARMADILLO_LIBRARIES})