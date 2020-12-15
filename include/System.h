#pragma once

#ifdef _WIN32
    #define NOMINMAX
    #include "Windows.h"
#elif defined __linux__
    #include "sys/sysinfo.h"
#endif

/**
 * @brief Get physical RAM usage
 * @return The physical RAM usage
 */
inline auto GetRamUsage()
{
#ifdef _WIN32
    MEMORYSTATUSEX memInfo{ sizeof(MEMORYSTATUSEX) };
    GlobalMemoryStatusEx(&memInfo);
    return memInfo.ullTotalPhys - memInfo.ullAvailPhys;
#elif defined __linux__
    struct sysinfo memInfo;
    sysinfo(&memInfo);
    return (memInfo.totalram - memInfo.freeram) * memInfo.mem_unit;
#endif
}