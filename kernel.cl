void kernel tabu_matrix_update(global int* matrix, const int size)
{  
    for(int i = 0; i + get_global_id(0) < size; i += get_global_size(0))
    {
        if (matrix[i+get_global_id(0)] > 0) matrix[i+get_global_id(0)]--;
    }
}

int get_tabu_matrix_value(global const int* matrix, const int vertex_count, const int pos, const int len, const int off)
{
    return matrix[pos * vertex_count * vertex_count + len * vertex_count + off];
}

int weight(global const int* adj_matrix, const int vertex_count, const int from, const int to)
{
    return adj_matrix[from * vertex_count + to];
}

int cost(global const int* adj_matrix, global const int* parent, const int vertex_count)
{
    int cost = weight(adj_matrix, vertex_count, parent[vertex_count-1], parent[0]);
    for (int i = 1; i < vertex_count; i++) cost += weight(adj_matrix, vertex_count, parent[i-1], parent[i]);
    return cost;
}

int mod(int value, int mod)
{
    return (value % mod + mod) % mod;
}

void place_solution(global int* solutions, global int* configs, const int idx, int val, int pos, int len, int off)
{
    solutions[idx] = val;
    configs[idx*3] = pos;
    configs[idx*3 + 1] = len;
    configs[idx*3 + 2] = off;
}

void kernel generate_rotates(global const int* adj_matrix, global const int* tabu_matrix, global const int* parent, global int* solutions, global int* configs, const int vertex_count)
{  
    int parent_cost = cost(adj_matrix, parent, vertex_count);

    {
        for (int pos_off = 0; pos_off + get_global_id(0) < vertex_count - 1; pos_off += get_global_size(0))
        {
            int pos = pos_off + get_global_id(0);
            for (int len_off = 2; pos + len_off + get_global_id(1) <= vertex_count && len_off + get_global_id(1) < vertex_count; len_off += get_global_size(1))
            {
                int len = len_off + get_global_id(1);
                for (int offset_o = 1; offset_o + get_global_id(2) < len; offset_o += get_global_size(2))
                {
                    int offset = offset_o + get_global_id(2);
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
                    else{
                        place_solution(solutions, configs, idx, 0, 0, 0, 0);
                    }
                }
            }
        }
    }
}

void kernel get_best1(global int* tabu_matrix, global const int* solutions, global const int* configs, const int vertex_count, global int* b_val, global int* b_pos, global int* b_len, global int* b_off)
{  
    const int cells = get_global_size(0);

    const int len = vertex_count*vertex_count*vertex_count;
    const int len_per_thrd = len/cells;
    
    if(get_global_id(0) < cells)
    {
        const int t = get_global_id(0);
        const int thrd_off = t * len_per_thrd;

        b_val[t] = -999999;
        b_len[t] = 0;

        for(int i=0; i + thrd_off + t < len && i <= len_per_thrd; i++)
        {
            int idx = i + thrd_off + t;

            int val = solutions[idx];
            int pos = configs[idx*3];
            int len = configs[idx*3 + 1];
            int off = configs[idx*3 + 2];

            if(len >= 2)
            {
                if(val > b_val[t])
                { 
                    b_val[t] = val;
                    b_pos[t] = pos;
                    b_len[t] = len;
                    b_off[t] = off;
                }
            }
        }
    }
}

void kernel get_best(global int* tabu_matrix, global const int* solutions, global const int* configs, const int vertex_count, global int* b_val, global int* b_pos, global int* b_len, global int* b_off, global int* ret)
{  
        // const int cells = 256;
        // if(get_global_id(0) == 0)
        // {
        //     int best_val = -999999;
        //     int best_pos = 0;
        //     int best_len = 0;
        //     int best_off = 0;

        //     for (int i=0; i < cells; i++)
        //     {
        //         int val = b_val[i];
        //         int pos = b_pos[i];
        //         int len = b_len[i];
        //         int off = b_off[i];
        //         if(val > best_val && len >= 2)
        //         {
        //             best_val = val;
        //             best_pos = pos;
        //             best_len = len;
        //             best_off = off;
        //         }
        //     }
        
        //     ret[0] = best_val;
        //     ret[1] = best_pos;
        //     ret[2] = best_len;
        //     ret[3] = best_off;

        //     if(best_len >= 2)
        //     tabu_matrix[best_pos * vertex_count * vertex_count + best_len * vertex_count + best_off] = vertex_count*vertex_count;
        // }
        // return;


    int len = 4096*4;


    for(int offset = 1; offset < len; offset *= 2)
    {
        for(int t_off = 0; t_off < len; t_off += get_global_size(0))
        {
            barrier(CLK_LOCAL_MEM_FENCE);

            int t = t_off + get_global_id(0);
            if(!(t & (2*offset - 1)))
            {
                int idx1 = t;
                int idx2 = idx1 + offset;

                if((b_len[idx1] < 2 && b_len[idx2] < 2) || b_len[idx2] < 2)
                {

                }
                else if(b_val[idx2] > b_val[idx1] || b_len[idx1] < 2)
                {
                    b_val[idx1] = b_val[idx2];
                    b_pos[idx1] = b_pos[idx2];
                    b_len[idx1] = b_len[idx2];
                    b_off[idx1] = b_off[idx2];
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(get_global_id(0) == 0)
    {
        ret[0] = b_val[0];
        ret[1] = b_pos[0];
        ret[2] = b_len[0];
        ret[3] = b_off[0];

        if(b_len[0] >= 2)
            tabu_matrix[b_pos[0] * vertex_count * vertex_count + b_len[0] * vertex_count + b_off[0]] = vertex_count*vertex_count;       
    }     
}

void kernel rotate(global int* path, const int pos, const int len, const int off)
{
    local int tmp[1024];

    if(get_group_id(0) == 0)
    {
        if(get_local_id(0) < len)
        {
            int tmp_idx = get_local_id(0);
            int path_idx = pos + (get_local_id(0) + off) % len;

            tmp[tmp_idx] = path[path_idx];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if(get_local_id(0) < len)
        {
            path[pos + get_local_id(0)] = tmp[get_local_id(0)];
        }
    }
}