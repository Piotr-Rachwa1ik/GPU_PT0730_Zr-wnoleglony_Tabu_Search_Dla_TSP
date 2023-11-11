cmake_minimum_required(VERSION 3.7)

project(tabu_tsp LANGUAGES CXX)

if(BUILD_CUDA_TABU)
    enable_language(CUDA)
    add_executable(main "main.cpp" "adjacency_matrix.cpp" "tabu.cpp" "tabu.cu")
    add_definitions(-DBUILD_CUDA_TABU)
    set_target_properties(main PROPERTIES
        CXX_STANDARD 20
        CXX_EXTENSIONS OFF
        CUDA_STANDARD 20
    )
else()
    add_executable(main "main.cpp" "adjacency_matrix.cpp" "tabu.cpp")
    set_target_properties(main PROPERTIES
        CXX_STANDARD 20
        CXX_EXTENSIONS OFF
    )
endif()
