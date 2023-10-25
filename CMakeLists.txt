cmake_minimum_required(VERSION 3.7)

project(tabu_tsp LANGUAGES CXX)

add_executable(main "main.cpp" "adjacency_matrix.cpp" "tabu.cpp")
set_target_properties(main PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
)
