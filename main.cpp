#include <iostream>
#include <array>
#include <functional>
#include "adjacency_matrix.h"
#include "tabu.h"


int main() 
{    
    Adjacency_matrix adjm;
    adjm.loadFromFile("tsp_171.txt");

    TSP_cuda_config cuda_cfg;
    std::chrono::seconds time_limit{1};

    const auto cuda_cfg_dialog = [&]{
        const auto read_xyz = [](TSP_cuda_config::xyz& xyz){
            std::cout << "x: ";
            std::cin >> xyz.x;
            std::cout << "y: ";
            std::cin >> xyz.y;
            std::cout << "z: ";
            std::cin >> xyz.z;
        };
        std::cout << "rotate_grid:\n";
        read_xyz(cuda_cfg.rotate_grid);
        std::cout << "rotate_block:\n";
        read_xyz(cuda_cfg.rotate_block);
        std::cout << "init_best_grid: ";
        std::cin >> cuda_cfg.init_best_grid;
        std::cout << "init_best_block: ";
        std::cin >> cuda_cfg.init_best_block;
        std::cout << "get_best_iter_grid: ";
        std::cin >> cuda_cfg.get_best_iter_grid;
        std::cout << "get_best_iter_block: ";
        std::cin >> cuda_cfg.get_best_iter_block;
    };

    const auto run = [&](auto&& policy){
        TSP_Tabu tabu{adjm, cuda_cfg};
        auto [path, cost] = tabu.solve(time_limit, policy);
        if (cost == 0) return;

        std::cout << "obliczony koszt: " << cost << "\n";
        std::cout << "obliczony cykl: ";
        for (int i = 0; i < path.size(); i++)
        {
            std::cout << path[i] << "->";
        }
        std::cout << path[0] << "\n";
    };

    const auto config_choices = std::to_array<std::pair<std::string_view, std::function<void()>>>({
        {"ustaw rozmiar grafu", [&]{
            std::cout << "podaj rozmiar: ";
            unsigned int size = 0;
            std::cin >> size;
            adjm.resize(size);
            adjm.generateRandom();
        }},
        {"wczytaj graf z pliku", [&]{
            std::cout << "podaj nazwe pliku: ";
            std::string fname;
            std::cin >> fname;
            adjm.loadFromFile(fname);
        }},
        {"ustaw czas", [&]{
            unsigned int t = 0;
            std::cin >> t;
            time_limit = std::chrono::seconds{t};
        }},
        {"konfiguracja cuda", 
            cuda_cfg_dialog
        }
    });


    const auto choices = std::to_array<std::pair<std::string_view, std::function<void()>>>({
        {"konfiguracja", [&]{
            for (auto i = 0;const auto& [name, choice] : config_choices)
            {
                std::cout << i << "." << name << "\n";
                i++;
            }
            std::cout << "wybor: ";
            unsigned int choice = 0;
            std::cin >> choice;
            if (choice > config_choices.size())
            {
                std::cout << "bledny wybor\n";
                return;
            }
            config_choices[choice].second();
        }},
        {"uruchom cpu jednowatkowe", [&]{
            run(TSP_Tabu::Exec_policy::cpu_single);
        }},
        {"uruchom cpu wielowatkowe", [&]{
            run(TSP_Tabu::Exec_policy::cpu_multi);
        }},
        {"uruchom cuda", [&]{
            run(TSP_Tabu::Exec_policy::cuda);
        }},
        {"zakoncz", [&]{
            std::exit(0);
        }}
    });

    for(;;)
    {
        std::cout << "========================\n";
        for (auto i = 0;const auto& [name, choice] : choices)
        {
            std::cout << i << "." << name << "\n";
            i++;
        }
        std::cout << "wybor: ";
        unsigned int choice = 0;
        std::cin >> choice;
        if (choice > choices.size())
        {
            std::cout << "bledny wybor\n";
            continue;
        }
        choices[choice].second();
    }
}
