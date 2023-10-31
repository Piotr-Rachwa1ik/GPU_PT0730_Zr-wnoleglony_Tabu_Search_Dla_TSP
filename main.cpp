#include <iostream>
#include <fstream>
#include "adjacency_matrix.h"
#include "tabu.h"

int main(int argc, char const *argv[]) 
{   
    if (argc < 3) return 1;

    auto filePath = argv[1];
    long maxIterations = strtol(argv[2], nullptr, 10);

    Adjacency_matrix adjm;
    adjm.loadFromFile(filePath);

    TSP_Tabu tabu{adjm};

    auto start = std::chrono::steady_clock::now();
    auto [path, cost] = tabu.solve(maxIterations);
    auto end = std::chrono::steady_clock::now();

    auto timeElapsed = end - start;
    if (argc == 3) {
        std::cout << "koszt: " << cost << "\n";
        std::cout << "cykl: ";

        for (int i = 0; i < path.size(); i++)
            std::cout << path[i] << "->";
        std::cout << path[0] << "\n";

        std::cout << "czas[ns]: " << timeElapsed.count() << "\n";
    }
    else {
        auto outputPath = argv[3];

        std::ofstream out{ outputPath };
        out << "koszt;" << cost << "\n";
        
        out << "cykl;";
        for (int i = 0; i < path.size(); i++)
            out << path[i] << "->";
        out << path[0] << "\n";

        out << "czas[ns]:" << timeElapsed.count();
    }
}