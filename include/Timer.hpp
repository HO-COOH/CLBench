/*****************************************************************//**
 * \file   Timer.hpp
 * \brief  A Simple scoped timer
 * 
 * \author Wenhao Li
 * \date   October 2020
 *********************************************************************/
#pragma once

#include <chrono>
#include <iostream>

template<bool printWhenDestructed = false>
class Timer
{
    std::chrono::steady_clock::time_point last;
public:
    Timer() : last{ std::chrono::steady_clock::now() }
    {
    }
    template<typename T>
    [[nodiscard]] long double perSec(T count) const
    {
        using FpSeconds = std::chrono::duration<long double, std::chrono::seconds::period>;
        return static_cast<long double>(count) / (FpSeconds(std::chrono::steady_clock::now() - last).count());
    }

    [[nodiscard]]auto getDuration() const
    {
        return std::chrono::steady_clock::now() - last;
    }

    [[nodiscard]]auto getTick() const
    {
        return getDuration().count();
    }

    ~Timer()
    {
        if constexpr (printWhenDestructed)
            std::cout << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - last).count() << " microsec\n";
    }
};