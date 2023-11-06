#pragma once
#include "adjacency_matrix.h"
#include <vector>
#include <chrono>
#include <utility>

struct TSP_result
{
    std::vector<unsigned int> cycle;
    int cost;
};

class Tabu_matrix
{
    std::vector<int> matrix;
    size_t m_size;

public:
    Tabu_matrix(size_t vertex_count) : m_size{vertex_count}
    {
        matrix.resize(vertex_count * vertex_count * vertex_count);
    }

    void update()
    {
        for (auto& i : matrix)
        {
            if (i > 0) i--;
        }
    }

    void clear()
    {
        for (auto& i : matrix)
        {
            i = 0;
        }
    }

    int& get(size_t pos, size_t len, size_t off)
    {
        return const_cast<int&>(std::as_const(*this).get(pos, len, off));
    }

    const int& get(size_t pos, size_t len, size_t off) const
    {
        return matrix[pos * m_size * m_size + len * m_size + off];
    }

    size_t size() const
    {
        return m_size*m_size*m_size;
    }
};

class TSP_Tabu
{
    Adjacency_matrix adjm;
    Tabu_matrix tabu_matrix;

    int cost(const std::vector<unsigned int>& path) const;
    int moveValue(int parent_cost, int neighbour_cost) const;

    struct Rotation_description { int pos, len, offset, move_value; };
    Rotation_description generateBestRotate(const std::vector<unsigned int>& parent) const;//generuje najlepsze parametry rotate dla danego cyklu parent

    void rotate(std::vector<unsigned int>& path, int pos, int len, unsigned int offset) const;

public:
    TSP_Tabu(const Adjacency_matrix& adjm) : adjm{adjm}, tabu_matrix{adjm.vertexCount()} {}

    TSP_result solve(const std::chrono::seconds time_limit);
};