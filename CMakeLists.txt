cmake_minimum_required(VERSION 3.7)

project(tabu_tsp LANGUAGES CUDA CXX)


add_executable(main "main.cpp" "adjacency_matrix.cpp" "tabu.cpp" "tabu.cu")
set_target_properties(main PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
    CUDA_STANDARD 20
)
