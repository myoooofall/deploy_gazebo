/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef LOOP_H
#define LOOP_H

#include <iostream>
#include <thread>
#include <chrono>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>
#include <sstream>
#include <iomanip>
#include <exception>
#include <cstdint>

class LoopFunc
{
public:
    LoopFunc(const std::string &name, double period, std::function<void()> func, int bindCPU = -1)
        : _name(name), _period(period), _func(func), _bindCPU(bindCPU), _running(false) {}

    void start()
    {
        _running = true;
        log("[Loop Start] named: " + _name + ", period: " + formatPeriod() + "(ms)" + (_bindCPU != -1 ? ", run at cpu: " + std::to_string(_bindCPU) : ", cpu unspecified"));
        if (_bindCPU != -1)
        {
            _thread = std::thread(&LoopFunc::loop, this);
            setThreadAffinity(_thread.native_handle(), _bindCPU);
        }
        else
        {
            _thread = std::thread(&LoopFunc::loop, this);
        }
    }

    void shutdown()
    {
        {
            std::unique_lock<std::mutex> lock(_mutex);
            _running = false;
            _cv.notify_one();
        }
        if (_thread.joinable())
        {
            _thread.join();
        }
        log("[Loop End] named: " + _name);
    }

private:
    std::string _name;
    double _period;
    std::function<void()> _func;
    int _bindCPU;
    std::atomic<bool> _running;
    std::mutex _mutex;
    std::condition_variable _cv;
    std::thread _thread;

    void loop()
    {
        const auto period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(_period));
        const auto statWindow = std::chrono::seconds(1);
        auto nextWakeup = std::chrono::steady_clock::now();
        auto windowStart = nextWakeup;
        uint64_t windowCycles = 0;
        uint64_t windowOverruns = 0;
        double windowMaxOverrunMs = 0.0;
        while (_running)
        {
            const auto cycleStart = std::chrono::steady_clock::now();
            try
            {
                _func();
            }
            catch (const std::exception &e)
            {
                log("[Loop Fatal] named: " + _name + ", exception: " + e.what());
                std::terminate();
            }
            catch (...)
            {
                log("[Loop Fatal] named: " + _name + ", unknown exception");
                std::terminate();
            }

            nextWakeup += period;
            const auto end = std::chrono::steady_clock::now();
            ++windowCycles;

            // Keep strict periodic pacing: if this cycle overruns, skip missed slots
            // instead of running back-to-back catch-up cycles.
            if (end > nextWakeup)
            {
                const auto lag = end - nextWakeup;
                const auto missed = static_cast<long long>(lag / period) + 1LL;
                nextWakeup += period * missed;
            }

            const auto execTime = end - cycleStart;
            if (execTime > period)
            {
                ++windowOverruns;
                const double overrunMs = std::chrono::duration<double, std::milli>(execTime - period).count();
                if (overrunMs > windowMaxOverrunMs)
                {
                    windowMaxOverrunMs = overrunMs;
                }
            }

            if (end - windowStart >= statWindow)
            {
                if (windowOverruns > 0)
                {
                    const double overrunRate = 100.0 * static_cast<double>(windowOverruns) / static_cast<double>(windowCycles);
                    std::ostringstream oss;
                    oss << "[Loop Overrun] named: " << _name
                        << ", period_ms: " << std::fixed << std::setprecision(2) << (_period * 1000.0)
                        << ", overruns: " << windowOverruns << "/" << windowCycles
                        << " (" << std::setprecision(1) << overrunRate << "%)"
                        << ", max_overrun_ms: " << std::setprecision(2) << windowMaxOverrunMs;
                    log(oss.str());
                }
                windowStart = end;
                windowCycles = 0;
                windowOverruns = 0;
                windowMaxOverrunMs = 0.0;
            }

            std::unique_lock<std::mutex> lock(_mutex);
            if (_cv.wait_until(lock, nextWakeup, [this]
                               { return !_running; }))
            {
                break;
            }
        }
    }

    std::string formatPeriod() const
    {
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(0) << _period * 1000;
        return stream.str();
    }

    void log(const std::string &message)
    {
        static std::mutex logMutex;
        std::lock_guard<std::mutex> lock(logMutex);
        std::cout << message << std::endl;
    }

    void setThreadAffinity(std::thread::native_handle_type threadHandle, int cpuId)
    {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpuId, &cpuset);
        if (pthread_setaffinity_np(threadHandle, sizeof(cpu_set_t), &cpuset) != 0)
        {
            std::ostringstream oss;
            oss << "Error setting thread affinity: CPU " << cpuId << " may not be valid or accessible.";
            throw std::runtime_error(oss.str());
        }
    }
};

#endif // LOOP_H
