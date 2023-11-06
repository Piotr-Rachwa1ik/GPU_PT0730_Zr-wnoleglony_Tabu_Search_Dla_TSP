#include <iostream>
#include "adjacency_matrix.h"
#include "tabu.h"


int main() 
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
