#pragma once
#include <iostream>
#include <algorithm>
#include <numeric>

namespace test::Benchmark::MatrixMultiplication
{
    struct Matrix
    {
        float* data = nullptr;
        size_t rows{};
        size_t columns{};
        bool noAlloc = false;
    public:
        struct NoAlloc {};
        Matrix() = default;
        Matrix(size_t row, size_t col, float value) : data(new float[row * col]), rows(row), columns(col) { std::fill(data, data + row * col, value); }
        Matrix(size_t row, size_t col) :data(new float[row * col]), rows(row), columns(col) {}
        Matrix(size_t row, size_t col, NoAlloc) : rows(row), columns(col), noAlloc(true) {}
        Matrix(Matrix&& m) noexcept : data(m.data), rows(m.rows), columns(m.columns)
        {
            m.data = nullptr;
#ifdef DEBUG
            puts("moved");
#endif
        }
        Matrix(Matrix const&) = delete;
        Matrix& operator=(Matrix&& m) = default;
        Matrix& operator=(Matrix const&) = delete;
        ~Matrix()
        {
            if (!noAlloc && data!=nullptr) 
                delete[] data;
        }

        [[nodiscard]]auto begin() { return data; }
        [[nodiscard]]auto end() { return data + rows * columns; }

        [[nodiscard]] float& operator()(size_t row, size_t col) { return data[col + row * columns]; }

        [[nodiscard]] const float& operator()(size_t row, size_t col) const { return data[col + row * columns]; }

        [[nodiscard]] Matrix transpose() const
        {
            Matrix temp{ columns, rows };
            for (size_t i = 0; i < rows; ++i)
            {
                for (size_t j = 0; j < columns; ++j)
                    temp(j, i) = (*this)(i, j);
            }
            return temp;
        }

        [[nodiscard]]auto size() const { return rows * columns; }
        [[nodiscard]]auto bytes() const { return sizeof(float) * size(); }

        static Matrix make_random_matrix(size_t row, size_t col)
        {
            Matrix m{ row, col };
            std::generate(m.data, m.data + row * col, [] { return static_cast<float>(rand()) / RAND_MAX; });
            return m;
        }
        static Matrix make_test_matrix(size_t row, size_t col)
        {
            Matrix m{ row, col };
            std::iota(m.begin(), m.end(), 0.0f);
            return m;
        }
        static Matrix make_debug_matrix(size_t row, size_t col)
        {
            return Matrix{ row, col, 1.0f };
        }
        friend std::ostream& operator<<(std::ostream& os, Matrix const& m)
        {
            os << '[';
            auto iter = m.data;
            for (size_t i = 0; i < m.rows; ++i)
            {
                os << '[';

                std::copy_n(iter, m.columns, std::ostream_iterator<float>{os, ", "});
                os << "]\n";
                iter += m.columns;
            }
            os << "]\n";
            return os;
        }

        class MatrixMultiplicationException : std::exception
        {
        public:
            const char* what() const noexcept override
            {
                return "Matrix size mismatch!";
            }
        };
    };
    void NaiveCPUMul(Matrix const& lhs, Matrix const& rhs)
    {
        Matrix result{ lhs.rows, rhs.columns };
        for(auto i=0; i<lhs.rows; ++i)
        {
            for(auto j=0; j<rhs.columns; ++j)
            {
                float sum{};
                for(auto k=0; k<lhs.columns; ++k)
                {
                    sum += lhs(i, k) * rhs(k, j);
                }
                result(i, j) = sum;
            }
        }
    }

    void TransposedCPUMul(Matrix const& lhs, Matrix const& transposedRhs)
    {
        Matrix result{ lhs.rows, transposedRhs.columns };
        for (auto i = 0; i < lhs.rows; ++i)
        {
            for (auto j = 0; j < transposedRhs.columns; ++j)
            {
                float sum{};
                for (auto k = 0; k < lhs.columns; ++k)
                {
                    sum += lhs(i, k) * transposedRhs(j, k);
                }
                result(i, j) = sum;
            }
        }
    }
}