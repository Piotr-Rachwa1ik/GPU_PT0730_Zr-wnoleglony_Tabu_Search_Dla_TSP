#include "tabu.h"
#include <random>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <random>
#include <type_traits>
#include <limits>


int TSP_Tabu::cost(const std::vector<unsigned int>& path) const
{
    int cost = adjm.weight(path.back(), path.front());
    for (int i = 1; i < path.size(); i++) cost += adjm.weight(path[i - 1], path[i]);
    return cost;
}

TSP_Tabu::Rotation_description TSP_Tabu::generateBestRotate(const std::vector<unsigned int>& parent) const
{
    int best_pos = 0;
    int best_len = 0;
    int best_offset = 0;
    int best_val = std::numeric_limits<int>::min();

    const auto parent_cost = cost(parent);

#pragma omp parallel for
    for (int pos = 0; pos < adjm.vertexCount() - 1; pos++)
    {
        for (int len = 2; pos + len <= adjm.vertexCount() && len < adjm.vertexCount(); len++)
        {
            for (int offset = 1; offset < len; offset++)
            {
                if (!tabu_matrix.get(pos, len, offset))
                {
                    auto mod = [mod = (int)adjm.vertexCount()](int value)
                    { 
                        return (value % mod + mod) % mod;
                    };

                    auto current_cost = parent_cost;

                
                    //odjecie wag krawedzi, ktore nie wystapia po rotacji
                    current_cost -= adjm.weight(parent[mod(pos - 1)], parent[pos]);
                    current_cost -= adjm.weight(parent[pos + offset - 1], parent[pos + offset]);
                    current_cost -= adjm.weight(parent[pos + len - 1], parent[mod(pos + len)]);       
                    

                    //dodanie wag krawedzi, ktore zostana dodane w wyniku rotacji
                    current_cost += adjm.weight(parent[mod(pos - 1)], parent[pos + offset]);
                    current_cost += adjm.weight(parent[pos + len - 1], parent[pos]);
                    current_cost += adjm.weight(parent[pos + offset - 1], parent[mod(pos + len)]);


#pragma omp critical 
                    if (moveValue(parent_cost, current_cost) > best_val)
                    {
                        best_val = moveValue(parent_cost, current_cost);
                        best_pos = pos;
                        best_len = len;
                        best_offset = offset;
                    }
                }
            }
        }
    }

    return { best_pos, best_len, best_offset, best_val };
}

void TSP_Tabu::rotate(std::vector<unsigned int>& path, int pos, int len, unsigned int offset) const
{
    auto begin_it = path.begin() + pos;
    auto end_it = begin_it + len;

    std::rotate(begin_it, begin_it + offset, end_it);
}

int TSP_Tabu::moveValue(int parent_cost, int neighbour_cost) const
{
    return parent_cost - neighbour_cost;
}


TSP_result TSP_Tabu::solve(const std::chrono::seconds time_limit)
{
    const auto start_t = std::chrono::steady_clock::now();

    tabu_matrix.clear();
    const auto rotate_tabu_cadency = adjm.vertexCount() * adjm.vertexCount();

    std::vector<unsigned int> current_solution;
    current_solution.resize(adjm.vertexCount());
    for (int i = 0; i < adjm.vertexCount(); i++) current_solution[i] = i;
    int current_cost = cost(current_solution);

    auto best_solution = current_solution;
    auto best_cost = current_cost;

    int it = 0;
    while (std::chrono::steady_clock::now() - start_t < time_limit)
    {
        auto [pos, len, offset, rot_value] = generateBestRotate(current_solution);
        rotate(current_solution, pos, len, offset);
        tabu_matrix.get(pos, len, offset) = rotate_tabu_cadency;

        current_cost = cost(current_solution);
        if (current_cost < best_cost)
        {
            best_cost = current_cost;
            best_solution = current_solution;
        }

        tabu_matrix.update();
        it++;
    }

    std::cout << "iter: " << it << "\n";

    return { std::move(best_solution), best_cost };
}
