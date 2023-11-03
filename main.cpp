#include <CL/cl.hpp>

#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <string_view>
#include <sstream>
#include <random>
#include <type_traits>

#include "adjacency_matrix.h"
#include "tabu.h"

#include "cl_utils.h"


template<typename T, typename Y>
std::common_type_t<T, Y> rand(T min, Y max)
{
    static std::mt19937 rng(std::random_device{}());

    using return_t = std::common_type_t<T, Y>;

    if constexpr (std::is_floating_point_v<return_t>)
    {
        return std::uniform_real_distribution<return_t>(min, max)(rng);
    }
    else
    {
        return std::uniform_int_distribution<return_t>(min, max)(rng);
    }
}

void bench(auto&& f)
{
    auto b = std::chrono::steady_clock::now();
    f();
    auto e = std::chrono::steady_clock::now();
    std::cout << "time: " << std::chrono::duration_cast<std::chrono::microseconds>(e-b).count() << "\n";
}

void tabu();

int main() 
{    
    tabu();
}


void tabu()
{
    Adjacency_matrix adjm;
    adjm.loadFromFile("tsp_171.txt");

    TSP_Tabu tabu{adjm};
    auto [path, cost] = tabu.solve(std::chrono::seconds{1});

    std::cout << "obliczony koszt: " << cost << "\n";
    std::cout << "obliczony cykl: ";
    for (int i = 0; i < path.size(); i++)
    {
        std::cout << path[i] << "->";
    }
    std::cout << path[0] << "\n";
}