// SPDX-License-Identifier: GPL-2.0-or-later
//
// NOT UPSTREAM. Written for nguo-3d.
//
// GCS.cpp logs solver diagnostics through FreeCAD's `Base::Console()`. Pulling
// in FreeCAD's Base library for a printf would be absurd, so this is a
// drop-in with the two methods planegcs actually calls (`log`, `warning`).
//
// Output is silenced by default: planegcs logs per-iteration in debug builds
// and we do not want a solver writing to stderr from inside a UI. Set
// NGUO_GCS_VERBOSE=1 in the environment to route it to stderr instead.
#pragma once

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

namespace Base
{

class ConsoleSingleton
{
public:
    void log(const char* fmt, ...) const
    {
        va_list ap;
        va_start(ap, fmt);
        emit(fmt, ap);
        va_end(ap);
    }

    void warning(const char* fmt, ...) const
    {
        va_list ap;
        va_start(ap, fmt);
        emit(fmt, ap);
        va_end(ap);
    }

private:
    static void emit(const char* fmt, va_list ap)
    {
        static const bool verbose = [] {
            const char* v = std::getenv("NGUO_GCS_VERBOSE");
            return v != nullptr && v[0] == '1';
        }();
        if (verbose) {
            std::vfprintf(stderr, fmt, ap);
        }
    }
};

inline ConsoleSingleton& Console()
{
    static ConsoleSingleton instance;
    return instance;
}

}  // namespace Base
