#include <iostream>
#include "adjacency_matrix.h"
#include "tabu.h"


int main() 
{    
    Adjacency_matrix adjm;
    adjm.loadFromFile("tsp_171.txt");
    // adjm.resize(200);
    // adjm.generateRandom();

    TSP_Tabu tabu{adjm};

    for (auto policy : {TSP_Tabu::Exec_policy::cpu_single, TSP_Tabu::Exec_policy::cpu_multi, TSP_Tabu::Exec_policy::cuda})
    {
        auto [path, cost] = tabu.solve(std::chrono::seconds{1}, policy);
        if (cost == 0) continue;

        std::cout << "obliczony koszt: " << cost << "\n";
        std::cout << "obliczony cykl: ";
        for (int i = 0; i < path.size(); i++)
        {
            std::cout << path[i] << "->";
        }
        std::cout << path[0] << "\n";
    }
}
