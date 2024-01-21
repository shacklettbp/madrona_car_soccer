#include <algorithm>
#include <stdio.h>
#include "physics.hpp"

namespace madEscape {

using namespace madrona::math;

struct Segment {
    Vector3 p1;
    Vector3 p2;
};

int intersectMovingSphereAABB(Sphere s,
                              Vector3 dx,
                              AABB aabb,
                              float &t)
{
    AABB e = aabb;

    e.pMin -= Vector3::all(s.radius);
    e.pMax += Vector3::all(s.radius);

    float t_out;
    bool intersected = e.rayIntersects(s.center, 
                                          Diag3x3::fromVec(dx).inv(),
                                          0.0f, 1.0f, t_out);

    t = t_out;

    return (int)intersected;
}

int intersectMovingOBBs2D(const OBB &a,
                          const OBB &b)
{
    // Simply loop through the normals of a, then b to find a separating axis.
    // Very stupid.
    auto find_sat = [](const OBB &a, const OBB &b) {
        static uint32_t min_max_lut[4][2] = {
            { 3, 0 },
            { 0, 1 },
            { 1, 2 },
            { 1, 0 }
        };

        for (int a_norm_idx = 0; a_norm_idx < 4; ++a_norm_idx) {
            Vector2 a_norm_perp = a.verts[(a_norm_idx + 1)%4] - a.verts[a_norm_idx];
            Vector2 a_norm = Vector2{ -a_norm_perp.y, a_norm_perp.x };
            a_norm /= a_norm.length();
            
            // Use the fact that on an OBB, along the normal axis, the min and max
            // are simply the two vertices we calculated the normal from.
            float a_min = a_norm.dot(a.verts[min_max_lut[a_norm_idx][0]]);
            float a_max = a_norm.dot(a.verts[min_max_lut[a_norm_idx][1]]);

            float b_min = FLT_MAX;
            float b_max = -FLT_MAX;

            for (int b_vert_idx = 0; b_vert_idx < 4; ++b_vert_idx) {
                const Vector2 &b_vert = b.verts[b_vert_idx];
                
                float b_dot_a_norm = b_vert.dot(a_norm);

                b_min = std::min(b_dot_a_norm, b_min);
                b_max = std::max(b_dot_a_norm, b_max);
            }

            if (a_max < b_min || a_min > b_max) {
                // Found a separating axis
                return 1;
            }
        }

        return 0;
    };

    if (find_sat(a, b)) {
        return 0;
    } else if (find_sat(b, a)) {
        return 0;
    }

    return 1;
}
    
}
