#include "tabu.h"
#include <random>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <random>
#include <type_traits>
#include <limits>
#include <bit>
#include <memory>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "tabu_kernels.cuh"



template<typename T>
struct Cuda_deleter
{
    void operator()(T* ptr)
    {
        cudaFree(ptr);
    }
};

template<typename T>
using cuda_uptr = std::unique_ptr<T, Cuda_deleter<T>>;

template<typename T>
cuda_uptr<T> makeCudaBuffer(size_t size)
{
    T* buf = nullptr;
    auto ret = cudaMalloc((void**)&buf, size*sizeof(T));
    return cuda_uptr<T>{buf};
}

template<typename T>
cuda_uptr<T> makeCudaBuffer(const std::vector<T>& src)
{
    return makeCudaBuffer<T>(src.size());
}

TSP_result TSP_Tabu::solve_cuda(const std::chrono::seconds time_limit, const TSP_cuda_config& cfg)
{
    std::vector<int> solutions(tabu_matrix.size());
    std::vector<int> configs(tabu_matrix.size() * 3);


    const auto start_t = std::chrono::steady_clock::now();

    tabu_matrix.clear();
    const auto rotate_tabu_cadency = adjm.vertexCount() * adjm.vertexCount();

    auto tabu_buf = makeCudaBuffer<int>(tabu_matrix.size());
    auto adjm_buf = makeCudaBuffer<int>(adjm.size());
    auto ret = cudaMemcpy(adjm_buf.get(), adjm.data(), adjm.size() * sizeof(int), cudaMemcpyHostToDevice);

    auto solutions_buf = makeCudaBuffer<int>(solutions.size());
    auto configs_buf = makeCudaBuffer<int>(configs.size());

    auto parent_buf = makeCudaBuffer<int>(adjm.vertexCount());
    cudaMemcpy(tabu_buf.get(), tabu_matrix.data(), tabu_matrix.size() * sizeof(int), cudaMemcpyHostToDevice);


    auto update_tabu = [&]
    {
        updateTabuMatrix<<<64, 1024>> >(tabu_buf.get(), tabu_matrix.size());
        cudaDeviceSynchronize();
    };

    std::vector<unsigned int> current_solution;
    current_solution.resize(adjm.vertexCount());
    for (int i = 0; i < adjm.vertexCount(); i++) current_solution[i] = i;
    int current_cost = cost(current_solution);

    auto best_solution = current_solution;
    auto best_cost = current_cost;

    std::vector<int> vals(solutions.size());
    std::vector<int> poss(solutions.size());
    std::vector<int> lens(solutions.size());
    std::vector<int> offs(solutions.size());

    auto vals_buf = makeCudaBuffer<int>(vals);
    auto poss_buf = makeCudaBuffer<int>(poss);
    auto lens_buf = makeCudaBuffer<int>(lens);
    auto offs_buf = makeCudaBuffer<int>(offs);

    cudaMemcpy(vals_buf.get(), vals.data(), vals.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(poss_buf.get(), poss.data(), poss.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(lens_buf.get(), lens.data(), lens.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(offs_buf.get(), offs.data(), offs.size() * sizeof(int), cudaMemcpyHostToDevice);

    int best[4] = {};
    auto best_buf = makeCudaBuffer<int>(4);
    cudaMemcpy(best_buf.get(), best, sizeof(best), cudaMemcpyHostToDevice);

    cudaMemcpy(solutions_buf.get(), solutions.data(), solutions.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(configs_buf.get(), configs.data(), configs.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(tabu_buf.get(), tabu_matrix.data(), tabu_matrix.size() * sizeof(int), cudaMemcpyHostToDevice);


    auto b_buf = makeCudaBuffer<int>(offs);
    auto b2_buf = makeCudaBuffer<int>(offs);


    auto generate_best_rotate = [&]()
    {
        cudaMemcpy(parent_buf.get(), current_solution.data(), current_solution.size() * sizeof(int), cudaMemcpyHostToDevice);

        generate_rotates<<<dim3(cfg.rotate_grid.x,cfg.rotate_grid.y,cfg.rotate_grid.z), dim3(cfg.rotate_block.x,cfg.rotate_block.y,cfg.rotate_block.z)>>>(adjm_buf.get(), tabu_buf.get(), parent_buf.get(), solutions_buf.get(), configs_buf.get(), adjm.vertexCount(), current_cost);
        init_best<<<cfg.init_best_grid, cfg.init_best_block>>>(b_buf.get(), solutions.size());
        get_best_iter<<<cfg.init_best_grid, cfg.init_best_block>>>(solutions.size(), solutions_buf.get(), configs_buf.get(), b_buf.get(), b2_buf.get());
        get_best_iter<<<1, cfg.init_best_block>>>(cfg.init_best_grid, solutions_buf.get(), configs_buf.get(), b2_buf.get(), b2_buf.get());
        get_best<<<1, 1>>>(tabu_buf.get(), current_solution.size(), solutions_buf.get(), configs_buf.get(), best_buf.get(), b2_buf.get());
        cudaDeviceSynchronize();

        cudaMemcpy(best, best_buf.get(), sizeof(best), cudaMemcpyDeviceToHost);

        return std::tuple{ best[1], best[2], best[3], best[0] };
    };

    int it= 0;
    while (std::chrono::steady_clock::now() - start_t < time_limit)
    {
        auto [pos, len, offset, rot_value] = generate_best_rotate();
        if(len >= 2)
        {
            rotate(current_solution, pos, len, offset);
            current_cost -= rot_value;
        }

        if (current_cost < best_cost)
        {
            best_cost = current_cost;
            best_solution = current_solution;
        }

        update_tabu();
        it++;
    }

    std::cout << "iter: " << it << "\n";

    return { std::move(best_solution), best_cost };
}