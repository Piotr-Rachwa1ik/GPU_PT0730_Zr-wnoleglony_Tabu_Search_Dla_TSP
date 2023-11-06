#include "adjacency_matrix.h"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <utility>
#include <random>



int rand(int min, int max)
{
    static std::mt19937 rng(std::random_device{}());
    return std::uniform_int_distribution<int>(min, max)(rng);
    //return std::clamp(std::normal_distribution<float>(50, 10)(rng), 0.f, 100.f);
}


void Adjacency_matrix::resize(size_t vertex_count)
{
    m_matrix.resize(vertex_count * vertex_count);
    m_size = vertex_count;
}

const Adjacency_matrix::weight_t& Adjacency_matrix::weight(size_t from, size_t to) const
{
    return m_matrix[from * m_size + to];
}

Adjacency_matrix::weight_t& Adjacency_matrix::weight(size_t from, size_t to)
{
    return const_cast<weight_t&>(std::as_const(*this).weight(from, to));
}

bool Adjacency_matrix::loadFromFile(const std::string& filename)
{
    std::ifstream in{ filename };
    if (in.is_open())
    {
        size_t m_size;
        in >> m_size;
        resize(m_size);

        if (!in.fail())
        {
            for (auto y = 0; y < m_size; y++)
            {
                for (auto x = 0; x < m_size; x++)
                {
                    weight_t w;
                    in >> w;
                    if (in.fail()) return false;
                    weight(y, x) = w;
                }
            }
        }
        else return false;
    }
    else return false;
    return true;
}

void Adjacency_matrix::generateRandom()
{
    for (int i = 0; i < vertexCount(); i++)
    {
        for (int j = 0; j < vertexCount(); j++)
        {
            if (i != j) weight(i, j) = rand(0, 100);
            else weight(i, j) = 0;
        }
    }
}

void Adjacency_matrix::saveToFile(const std::string& filename)
{
    std::ofstream os{ filename };
    os << vertexCount() << "\n";
    for (int i = 0; i < vertexCount(); i++)
    {
        for (int j = 0; j < vertexCount(); j++)
        {
            os << weight(i, j) << " ";
        }
        os << "\n";
    }

}

size_t Adjacency_matrix::vertexCount() const
{
    return m_size;  
}

void Adjacency_matrix::print() const
{
    constexpr auto field_width = 4;

    std::cout << std::setw(field_width + 1) << "";
    for (auto i = 0; i < vertexCount(); i++) std::cout << std::setw(field_width) << i;
    std::cout << "\n" << std::setw(field_width) << "";
    for (auto i = 0; i < vertexCount() * field_width + 1; i++) std::cout << "-";
    std::cout << '\n';

    for (auto x = 0; x < vertexCount(); x++)
    {
        std::cout << std::setw(field_width) << x << "|";
        for (auto y = 0; y < vertexCount(); y++)
        {
            std::cout << std::setw(field_width) << weight(x, y);
        }
        std::cout << '\n';
    }
}
