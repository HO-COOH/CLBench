#pragma once
#include <random>
#include <array>
namespace test::Benchmark::Convolution
{
    static std::mt19937 rdEng{ std::random_device{}() };
    struct NoAlloc {};
    template<int channels>
    struct Image
    {
        unsigned char* data = nullptr;
        size_t rows{};
        size_t columns{};
        bool noAlloc = false;

        Image() = default;
        Image(size_t row, size_t col) : data(new unsigned char[row * col*channels]), rows(row), columns(col) { }
        Image(size_t row, size_t col, NoAlloc) :data(nullptr), rows(row), columns(col), noAlloc(true) {}
        Image(Image&& rhs) noexcept : data(rhs.data), rows(rhs.rows), columns(rhs.columns)
        {
            rhs.data = nullptr;
#ifdef DEBUG
            puts("moved");
#endif
        }
        Image(Image const&) = delete;
        Image& operator=(Image&& m) = default;
        Image& operator=(Image const&) = delete;
        ~Image() { if (!noAlloc || data!=nullptr) delete[] data; }

        [[nodiscard]] auto begin() { return data; }
        [[nodiscard]] auto end() { return data + rows * columns*channels; }

        [[nodiscard]] auto& operator()(size_t row, size_t col, int channel) { return data[(col + row * columns)*channels+channel-1]; }

        [[nodiscard]] auto operator()(size_t row, size_t col, int channel) const { return data[(col + row * columns) * channels + channel-1]; }

        [[nodiscard]] auto size() const { return rows * columns * channels; }
    };

    template<int FS, int channels>
    struct Filter
    {
        std::array<float, FS * FS * channels> data;
        [[nodiscard]] auto& operator()(int row, int col, int channel) { return data[(col + row * FS) * channels + channel-1]; }
        [[nodiscard]] auto operator()(int row, int col, int channel) const { return data[(col + row * FS) * channels + channel-1]; }
        [[nodiscard]] static constexpr auto halfSize() { return FS / 2; }
        [[nodiscard]] static constexpr auto size() { return FS; }
        [[nodiscard]] static constexpr auto area() { return FS * FS; }
        [[nodiscard]] static constexpr auto channel() { return channels; }

        static auto makeFilter()
        {
            Filter f;
            std::generate(std::begin(f.data), std::end(f.data), [dist = std::normal_distribution<float>{ 0 }]()mutable{ return dist(rdEng); });
            return f;
        }
    };

    template<int channels, int FS>
    void NaiveCPU(Image<channels> const& image, Filter<FS, channels> const& filter)
    {
        const auto hfs = filter.halfSize();
        Image<channels> result{image.rows, image.columns};
        for(int i=0+hfs; i<(int)image.rows-hfs; ++i)
        {
            for(int j = 0 + hfs; j < (int)image.columns - hfs; ++j)
            {
                float sum{};
                for(int m=i-hfs; m<i+hfs; ++m)
                {
                    for(int n=j-hfs; n<j+hfs; ++n)
                    {
                        sum += image(i, j, channels) * filter(m-i + hfs, n-j + hfs, channels);
                    }
                }
                result(i, j, channels) = sum;
            }
        }
    }
}