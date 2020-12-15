#include "IO.hpp"
#include "GPU.h"
#include "Test.hpp"

int main()
{
    test::DataTransfer::DataTransfer();
    test::Compilation::Compilation();
    test::Benchmark::Reduction::Reduction();
    test::Benchmark::Convolution::Convolution();
    test::Benchmark::Convolution::Convolution();
    std::cout << "\aFinished all testing!";
}
