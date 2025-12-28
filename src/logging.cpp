/*******************************************************************************
    Copyright (c) 2021, Andrea Maggiordomo, Paolo Cignoni and Marco Tarini

    This file is part of TextureDefrag, a reference implementation for
    the paper ``Texture Defragmentation for Photo-Reconstructed 3D Models''
    by Andrea Maggiordomo, Paolo Cignoni and Marco Tarini.

    TextureDefrag is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    TextureDefrag is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with TextureDefrag. If not, see <https://www.gnu.org/licenses/>.
*******************************************************************************/

#include "logging.h"

#include <string>
#include <fstream>
#include <sstream>
#include <thread>
#include <mutex>
#include <map>
#include <iomanip>
#include <iostream>

#ifdef __APPLE__
#include <mach/mach.h>
#endif

namespace logging {

Buffer::Buffer(int level)
    : os{}
{
    switch(level) {
    case -2:
        os << std::setw(8);
        os << " ERR| ";
        break;
    case -1:
        os << std::setw(8);
        os << "WARN| ";
        break;
    default:
        os << std::setw(6);
        os << level << "| ";
    }
}

Buffer::~Buffer()
{
    Logger::Log(os.str());
}

int Logger::logLevel = 0;
std::vector<std::ostream *> Logger::streamVec{};
std::map<std::thread::id, std::string> Logger::threadNames{};
std::mutex Logger::singletonMtx{};

void Logger::Init(int level)
{
    Logger::logLevel = level;
    threadNames[std::this_thread::get_id()] = "MainThread";
}

int Logger::GetLogLevel()
{
    return Logger::logLevel;
}

void Logger::RegisterStream(std::ostream *os)
{
    std::lock_guard<std::mutex> lock{Logger::singletonMtx};
    Logger::streamVec.push_back(os);
}

void Logger::RegisterName(const std::string& threadName)
{
    std::lock_guard<std::mutex> lock{Logger::singletonMtx};
    threadNames[std::this_thread::get_id()] = threadName;
}

std::string Logger::GetName()
{
    std::lock_guard<std::mutex> lock{Logger::singletonMtx};
    auto tid = std::this_thread::get_id();
    if (threadNames.count(tid) > 0)
        return threadNames[tid];
    else {
        std::stringstream ss;
        ss << tid;
        return ss.str();
    }
}

void Logger::Log(const std::string& s)
{
    std::stringstream ss;
    ss << std::setw(16) << Logger::GetName() << " | " << s << std::endl;

    std::lock_guard<std::mutex> lock{Logger::singletonMtx};

    std::cout << ss.str() << std::flush;

    for (auto os : streamVec)
        (*os) << ss.str() << std::flush;
}

void LogMemoryUsage()
{
#ifdef __linux__
    std::ifstream status("/proc/self/status");
    if (!status.is_open()) {
        LOG_WARN << "[MEM] Could not open /proc/self/status";
        return;
    }
    std::string line;
    while (getline(status, line)) {
        if (line.rfind("VmSize:", 0) == 0 || line.rfind("VmRSS:", 0) == 0 || line.rfind("VmHWM:", 0) == 0) {
            LOG_INFO << "[MEM] " << line;
        }
    }
#elif defined(__APPLE__)
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) == KERN_SUCCESS) {
        LOG_INFO << "[MEM] RSS: " << (info.resident_size / (1024 * 1024)) << " MB";
    }
#else
    LOG_WARN << "LogMemoryUsage() not implemented for this platform.";
#endif
}

} // namespace logging
