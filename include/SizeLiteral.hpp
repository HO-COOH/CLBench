/*****************************************************************//**
 * \file   SizeLiteral.hpp
 * \brief  Frequently-used memory size literals
 * 
 * \author Wenhao Li
 * \date   October 2020
 *********************************************************************/
#pragma once

constexpr unsigned long long operator"" _kb(unsigned long long kb)
{
    return kb * 1024;
}

constexpr unsigned long long operator"" _mb(unsigned long long mb)
{
    return mb * 1024 * 1024;
}

constexpr unsigned long long operator"" _gb(unsigned long long gb)
{
    return gb * 1024 * 1024 * 1024;
}

constexpr double toMb(unsigned long long bytes)
{
    return bytes / 1024.0 / 1024.0;
}

constexpr double toKb(unsigned long long bytes)
{
    return bytes / 1024.0;
}

constexpr double toGb(unsigned long long bytes)
{
    return bytes / 1024.0 / 1024.0 / 1024.0;
}