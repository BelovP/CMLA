include_directories(${PROJECT_SOURCE_DIR}/yalal)

foreach(PACKAGE ${PACKAGES})
    add_executable(test_${PACKAGE} test_${PACKAGE}.cpp)
    target_link_libraries(test_${PACKAGE} yalal)
endforeach()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/run_all_tests.sh
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})