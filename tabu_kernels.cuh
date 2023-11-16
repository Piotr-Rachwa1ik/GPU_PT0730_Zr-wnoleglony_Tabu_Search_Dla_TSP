#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>

__global__ void updateTabuMatrix(int* matrix, const int size)
{
    for (int i = 0; i + threadIdx.x < size; i += blockDim.x * gridDim.x)
    {
        if (matrix[i + blockDim.x * gridDim.x] > 0) matrix[i + blockDim.x * gridDim.x]--;
    }
}

__device__ int get_tabu_matrix_value(const int* matrix, const int vertex_count, const int pos, const int len, const int off)
{
    return matrix[pos * vertex_count * vertex_count + len * vertex_count + off];
}

__device__ int weight(const int* adj_matrix, const int vertex_count, const int from, const int to)
{
    return adj_matrix[from * vertex_count + to];
}

__device__ int cost(const int* adj_matrix, const int* parent, const int vertex_count)
{
    int cost = weight(adj_matrix, vertex_count, parent[vertex_count - 1], parent[0]);
    for (int i = 1; i < vertex_count; i++) cost += weight(adj_matrix, vertex_count, parent[i - 1], parent[i]);
    return cost;
}

__device__ int mod(int value, int mod)
{
    return (value % mod + mod) % mod;
}

__device__ void place_solution(int* solutions, int* configs, const int idx, int val, int pos, int len, int off)
{
    solutions[idx] = val;
    configs[idx * 3] = pos;
    configs[idx * 3 + 1] = len;
    configs[idx * 3 + 2] = off;
}

__global__ void generate_rotates(const int* adj_matrix, const int* tabu_matrix, const int* parent, int* solutions, int* configs, const int vertex_count, const int parent_cost)
{
    {
        int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
        int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
        int idx_z = blockDim.z * blockIdx.z + threadIdx.z;
        for (int pos_off = 0; pos_off + idx_x < vertex_count - 1; pos_off += blockDim.x * gridDim.x)
        {
            int pos = pos_off + idx_x;
            for (int len_off = 2; pos + len_off + idx_y <= vertex_count && len_off + idx_y < vertex_count; len_off += blockDim.y * gridDim.y)
            {
                int len = len_off + idx_y;
                for (int offset_o = 1; offset_o + idx_z < len; offset_o += blockDim.z * gridDim.z)
                {
                    int offset = offset_o + idx_z;
                    
                    // printf("pos: %d, len: %d, off: %d\n", pos, len, offset);
                    int idx = pos * vertex_count * vertex_count + len * vertex_count + offset;
                    if (get_tabu_matrix_value(tabu_matrix, vertex_count, pos, len, offset) == 0)
                    {
                        int current_cost = parent_cost;


                        //odjecie wag krawedzi, ktore nie wystapia po rotacji
                        current_cost -= weight(adj_matrix, vertex_count, parent[mod(pos - 1, vertex_count)], parent[pos]);
                        current_cost -= weight(adj_matrix, vertex_count, parent[pos + offset - 1], parent[pos + offset]);
                        current_cost -= weight(adj_matrix, vertex_count, parent[pos + len - 1], parent[mod(pos + len, vertex_count)]);


                        //dodanie wag krawedzi, ktore zostana dodane w wyniku rotacji
                        current_cost += weight(adj_matrix, vertex_count, parent[mod(pos - 1, vertex_count)], parent[pos + offset]);
                        current_cost += weight(adj_matrix, vertex_count, parent[pos + len - 1], parent[pos]);
                        current_cost += weight(adj_matrix, vertex_count, parent[pos + offset - 1], parent[mod(pos + len, vertex_count)]);



                        place_solution(solutions, configs, idx, parent_cost - current_cost, pos, len, offset);
                    }
                    else {
                        place_solution(solutions, configs, idx, 0, 0, 0, 0);
                    }
                }
            }
        }
    }
}

__global__ void init_best(int* b, int len)
{
    for (int i = 0; i < len; i += blockDim.x * gridDim.x)
    {
        int x = i + blockDim.x * blockIdx.x + threadIdx.x;
        b[x] = x;
    }
}


__global__ void get_best_iter(const int len, int* solutions, int* configs, const int* best, int* g_buf)
{
    int t = threadIdx.x;
    int gt = t + blockIdx.x * blockDim.x;
    int best_idx = best[gt];
    for (int i = gt; i < len; i += blockDim.x * gridDim.x)
    {
        int idx = i;
        int best_len = configs[best_idx * 3 + 1];
        int len2 = configs[idx * 3 + 1];
        int best_val = solutions[best_idx];
        int val2 = solutions[idx];

        if (len2 >= 2 && (val2 > best_val || best_len < 2))
        {
            best_idx = idx;
        }
    }

    __shared__ int best_local[1024];
    best_local[t] = best_idx;
    __syncthreads();


    for (int offset = 1; offset < 1024; offset *= 2) 
    {
        if (t % (2*offset) == 0)
    //for (int offset = blockDim.x / 2; offset > 0; offset /= 2)
    //{
    //    if (t < offset)
        {
            int bidx1 = t;
            int bidx2 = bidx1 + offset;
            int idx1 = best_local[bidx1];
            int idx2 = best_local[bidx2];

            int len1 = configs[idx1 * 3 + 1];
            int len2 = configs[idx2 * 3 + 1];
            int val1 = solutions[idx1];
            int val2 = solutions[idx2];

            if (len2 >= 2 && (val2 > val1 || len1 < 2))
            {
                best_local[bidx1] = best_local[bidx2];
            }
        }
        __syncthreads();
    }
    if (t == 0) g_buf[blockIdx.x] = best_local[0];
}

__global__ void get_best(int* tabu_matrix, const int vertex_count, int* solutions, int* configs, int* ret, int* best)
{
    if (blockDim.x * blockIdx.x + threadIdx.x == 0)
    {
        int val = solutions[best[0]];
        int pos = configs[best[0] * 3];
        int len = configs[best[0] * 3 + 1];
        int off = configs[best[0] * 3 + 2];


        ret[0] = val;
        ret[1] = pos;
        ret[2] = len;
        ret[3] = off;

        if (ret[2] >= 2)
            tabu_matrix[ret[1] * vertex_count * vertex_count + ret[2] * vertex_count + ret[3]] = vertex_count * vertex_count;
    }
}

__global__ void rotate2(int* path, const int pos, const int len, const int off)
{
    __shared__ int tmp[1024];

    //if (get_group_id(0) == 0)
    {
        if (blockDim.x * blockIdx.x + threadIdx.x < len)
        {
            int tmp_idx = blockDim.x * blockIdx.x + threadIdx.x;
            int path_idx = pos + (blockDim.x * blockIdx.x + threadIdx.x + off) % len;

            tmp[tmp_idx] = path[path_idx];
        }
        __syncthreads();
        if (blockDim.x * blockIdx.x + threadIdx.x < len)
        {
            path[pos + blockDim.x * blockIdx.x + threadIdx.x] = tmp[blockDim.x * blockIdx.x + threadIdx.x];
        }
    }
}