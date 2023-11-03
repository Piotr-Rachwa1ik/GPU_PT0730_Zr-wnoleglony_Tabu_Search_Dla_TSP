cmake_minimum_required(VERSION 3.7)

project(opencl_cmake VERSION 0.0.1 LANGUAGES CXX)

add_executable(main "main.cpp" "adjacency_matrix.cpp" "tabu.cpp")
set_target_properties(main PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
)

find_package(OpenCL REQUIRED)
target_link_libraries(main OpenCL::OpenCL)

add_custom_command(TARGET main POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PROJECT_SOURCE_DIR}/kernel.cl"
        $<TARGET_FILE_DIR:main>)

add_custom_command(TARGET main POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PROJECT_SOURCE_DIR}/tsp_17.txt"
        $<TARGET_FILE_DIR:main>)

add_custom_command(TARGET main POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PROJECT_SOURCE_DIR}/tsp_171.txt"
        $<TARGET_FILE_DIR:main>)

add_custom_command(TARGET main POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${PROJECT_SOURCE_DIR}/tsp_443.txt"
        $<TARGET_FILE_DIR:main>)
