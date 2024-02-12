#pragma once

#include <cstdint>

namespace madEscape {

enum class SimFlags : uint32_t {
    Default                = 0,
    StaggerStarts          = 1 << 1,
};

inline SimFlags & operator|=(SimFlags &a, SimFlags b);
inline SimFlags operator|(SimFlags a, SimFlags b);
inline SimFlags & operator&=(SimFlags &a, SimFlags b);
inline SimFlags operator&(SimFlags a, SimFlags b);

}

#include "sim_flags.inl"
