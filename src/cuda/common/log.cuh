#pragma once
#include <cstdlib>
#include <cstdio>

static inline bool sw_verbose() {
    static int v = -1;
    if (v < 0) { const char* e = getenv("VERBOSE"); v = (e && e[0] == '1') ? 1 : 0; }
    return v != 0;
}
#define SW_LOG(...) do { if (sw_verbose()) fprintf(stderr, __VA_ARGS__); } while(0)
