#pragma once
#include <vector>
#include <string>

class Adjacency_matrix
{
    using weight_t = int;

    std::vector<weight_t> m_matrix;
    unsigned int m_size = 0;

public:
    bool loadFromFile(const std::string& filename);
    void generateRandom();

    void resize(size_t vertex_count);

    const weight_t& weight(size_t from, size_t to) const;
    weight_t& weight(size_t from, size_t to);

    void saveToFile(const std::string& filename);

    size_t vertexCount() const;

    void print() const;

    weight_t* data()
    {
        return m_matrix.data();
    }
    size_t size() const
    {
        return m_size * m_size;
    }
};
