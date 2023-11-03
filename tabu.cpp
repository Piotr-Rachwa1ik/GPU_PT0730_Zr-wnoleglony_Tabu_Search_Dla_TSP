#include "tabu.h"
#include <random>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <random>
#include <type_traits>
#include <limits>

#include <CL/cl.hpp>
#include "cl_utils.h"

#undef min


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
    auto devices = list_devices();
    auto& default_device = devices[0];
    cl::Context context({default_device});

    std::vector<int> solutions(tabu_matrix.size());
    std::vector<int> configs(tabu_matrix.size() * 3);


    cl::Buffer tabu_buf(context, CL_MEM_READ_WRITE, tabu_matrix.size()*sizeof(int));
    cl::Buffer adjm_buf(context, CL_MEM_READ_WRITE, adjm.size()*sizeof(int));

    cl::Buffer solutions_buf(context, CL_MEM_READ_WRITE, solutions.size()*sizeof(int));
    cl::Buffer configs_buf(context, CL_MEM_READ_WRITE, configs.size()*sizeof(int));

    cl::CommandQueue queue(context, default_device);
    
    auto kernel_code = load_kernel("kernel.cl");
    cl::Program::Sources sources;
    sources.push_back({ kernel_code.c_str(), kernel_code.length() });

    cl::Program program(context, sources);
    if (program.build({ default_device }) != CL_SUCCESS) {
        std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << "\n";
        exit(1);
    }

    cl::make_kernel<cl::Buffer, int> tabu_matrix_update(cl::Kernel(program, "tabu_matrix_update"));
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> get_best(cl::Kernel(program, "get_best"));
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, int, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer> get_best1(cl::Kernel(program, "get_best1"));
    cl::make_kernel<cl::Buffer, int, int, int> rotate_kernel(cl::Kernel(program, "rotate"));
    cl::make_kernel<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, int> generate_rotates(cl::Kernel(program, "generate_rotates"));


    const auto start_t = std::chrono::steady_clock::now();

    tabu_matrix.clear();
    int iterations_without_improvement = 0;

    const auto rotate_tabu_cadency = adjm.vertexCount() * adjm.vertexCount();


    std::vector<unsigned int> current_solution;
    current_solution.resize(adjm.vertexCount());
    for (int i = 0; i < adjm.vertexCount(); i++) current_solution[i] = i;
    int current_cost = cost(current_solution);

    auto best_solution = current_solution;
    auto best_cost = current_cost;

    cl::Buffer parent_buf(context, CL_MEM_READ_WRITE, current_solution.size()*sizeof(int));

    int best[4] = {};
    cl::Buffer best_buf(context, CL_MEM_WRITE_ONLY, 4*sizeof(int));

    queue.enqueueWriteBuffer(tabu_buf, CL_TRUE, 0, tabu_matrix.size()*sizeof(int), &tabu_matrix.get(0,0,0));
    queue.enqueueWriteBuffer(adjm_buf, CL_TRUE, 0, adjm.size()*sizeof(int), adjm.data());
    queue.enqueueWriteBuffer(parent_buf, CL_TRUE, 0, current_solution.size()*sizeof(int), current_solution.data());
    queue.enqueueWriteBuffer(configs_buf, CL_TRUE, 0, configs.size()*sizeof(int), configs.data());
    queue.enqueueWriteBuffer(solutions_buf, CL_TRUE, 0, solutions.size()*sizeof(int), solutions.data());

    std::vector<int> vals(solutions.size());
    std::vector<int> poss(solutions.size());
    std::vector<int> lens(solutions.size());
    std::vector<int> offs(solutions.size());

    cl::Buffer vals_buf(context, CL_MEM_READ_WRITE, vals.size()*sizeof(int));
    cl::Buffer poss_buf(context, CL_MEM_READ_WRITE, poss.size()*sizeof(int));
    cl::Buffer lens_buf(context, CL_MEM_READ_WRITE, lens.size()*sizeof(int));
    cl::Buffer offs_buf(context, CL_MEM_READ_WRITE, offs.size()*sizeof(int));

    queue.enqueueWriteBuffer(vals_buf, CL_TRUE, 0, vals.size()*sizeof(int), vals.data());
    queue.enqueueWriteBuffer(poss_buf, CL_TRUE, 0, poss.size()*sizeof(int), poss.data());
    queue.enqueueWriteBuffer(lens_buf, CL_TRUE, 0, lens.size()*sizeof(int), lens.data());
    queue.enqueueWriteBuffer(offs_buf, CL_TRUE, 0, offs.size()*sizeof(int), offs.data());

    bool profile = false;
    auto launch_profile = [&](auto&& f, const std::string& name){
        if(profile)
        {
            auto b = std::chrono::steady_clock::now();
            f();
            auto e = std::chrono::steady_clock::now();
            std::cout << name << " " << std::chrono::duration_cast<std::chrono::microseconds>(e-b).count() << "us\n";
        }
        else f();
    };

    auto generate_best_rotate = [&]() {
        launch_profile([&]{
            generate_rotates(cl::EnqueueArgs(queue, cl::NDRange(adjm.vertexCount(), adjm.vertexCount(),adjm.vertexCount())), adjm_buf, tabu_buf, parent_buf, solutions_buf, configs_buf, current_solution.size()).wait();
        }, "gen_rotates");
        launch_profile([&]{
            get_best1(cl::EnqueueArgs(queue, cl::NDRange(4096*4)), tabu_buf, solutions_buf, configs_buf, current_solution.size(), vals_buf, poss_buf, lens_buf, offs_buf).wait();
        }, "best1");
        launch_profile([&]{
            get_best(cl::EnqueueArgs(queue, cl::NDRange(1024), cl::NDRange(1024)), tabu_buf, solutions_buf, configs_buf, current_solution.size(), vals_buf, poss_buf, lens_buf, offs_buf, best_buf).wait();
        }, "best");
        queue.enqueueReadBuffer(best_buf, CL_TRUE, 0, 4*sizeof(int), best);
        
        return std::tuple{best[1], best[2], best[3], best[0]};
    };






    int it= 0;
    while (std::chrono::steady_clock::now() - start_t < time_limit)
    {
        // queue.enqueueReadBuffer(tabu_buf, CL_TRUE, 0, tabu_matrix.size()*sizeof(int), &tabu_matrix.get(0,0,0));
        // auto [pos, len, offset, rot_value] = generateBestRotate(current_solution);
        // auto [pos2, len2, offset2, rot_value2] = generate_best_rotate();
        auto [pos, len, offset, rot_value] = generate_best_rotate();
        // if(pos != pos2 || len != len2 || offset != offset2 || rot_value != rot_value2)
        // {
        //     std::cout << pos << " " << len << " " << offset << " " << rot_value << "\n";
        //     std::cout << pos2 << " " << len2 << " " << offset2 << " " << rot_value2 << "\n";
        //     for(;;){}
        // }
        if(len >= 2)
        {
            // std::cout << pos << " " << len << " " << offset << " " << rot_value << "\n";
            // rotate(current_solution, pos, len, offset);
            launch_profile([&]{
                rotate_kernel(cl::EnqueueArgs(queue, cl::NDRange(adjm.vertexCount())), parent_buf, pos, len, offset).wait();
            }, "rotate");
            // tabu_matrix.get(pos, len, offset) = rotate_tabu_cadency;
            current_cost -= rot_value;
        }

        if (current_cost < best_cost)
        {
            best_cost = current_cost;
            queue.enqueueReadBuffer(parent_buf, CL_TRUE, 0, best_solution.size()*sizeof(int), best_solution.data());
            // best_solution = current_solution;
        }

        // tabu_matrix.update();
        launch_profile([&]{
            tabu_matrix_update(cl::EnqueueArgs(queue, cl::NDRange(1024*1024)), tabu_buf, tabu_matrix.size()).wait();
        }, "tabu_update");
        it++;
    }

    std::cout << "iter: " << it << "\n";

    return { std::move(best_solution), best_cost };
}
