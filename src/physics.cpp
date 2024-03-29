#include <algorithm>
#include <stdio.h>
#include <assert.h>
#include "physics.hpp"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

namespace madEscape {

using namespace madrona::math;

struct Segment {
    Vector3 p1;
    Vector3 p2;
};

int intersectMovingSphereAABB(Sphere s,
                              Vector3 dx,
                              AABB aabb,
                              float &t,
                              Vector3 &s_pos_out)
{
    AABB e = aabb;

    e.pMin -= Vector3::all(s.radius);
    e.pMax += Vector3::all(s.radius);

    float t_out;
    bool intersected = e.rayIntersects(s.center, 
                                          Diag3x3::fromVec(dx).inv(),
                                          0.0f, 1.0f, t_out);

    t = t_out;

#if 1
    if (s.center.x >= aabb.pMin.x && s.center.x <= aabb.pMax.x &&
        s.center.y >= aabb.pMin.y && s.center.y <= aabb.pMax.y) {

        // We need to push the car out of the sphere no matter what
        Vector3 car_to_ball = s.center - (aabb.pMin + aabb.pMax) * 0.5f;
        Vector3 aabb_dim = (aabb.pMax - aabb.pMin) * 0.5f;

        Vector2 car_to_ball_norm = { car_to_ball.x / aabb_dim.x,
                                     car_to_ball.y / aabb_dim.y };

        int largest_mag_idx = 
            abs(car_to_ball_norm.x) > abs(car_to_ball_norm.y) ? 0 : 1;

        float comp_radius[2] = {
            0.5f * (aabb.pMax.x - aabb.pMin.x),
            0.5f * (aabb.pMax.y - aabb.pMin.y),
        };

        float car_to_ball_dx[2] = {
            car_to_ball.x, car_to_ball.y
        };

        Vector3 ball_new_pos = s.center + 
            abs(comp_radius[largest_mag_idx] / car_to_ball_dx[largest_mag_idx]) * 
            car_to_ball;

        s_pos_out = ball_new_pos;

        return 1;
    }
#endif

    // Update the sphere's position in the case of a collision so that the sphere
    // is no longer colliding with the OBB
    s_pos_out = s.center + t * s.center;

    return (int)intersected;
}

int intersectSphereAABB(Sphere s,
                       const AABB &box,
                       Vector3 &overlap)
{
    if (s.center.x > box.pMin.x && s.center.x < box.pMax.x &&
        s.center.y > box.pMin.y && s.center.y < box.pMax.y) {
        printf("WRONG\n");
    }

    Vector3 closest = {
        MAX(box.pMin.x, MIN(s.center.x, box.pMax.x)),
        MAX(box.pMin.y, MIN(s.center.y, box.pMax.y)),
        MAX(box.pMin.z, MIN(s.center.z, box.pMax.z)),
    };

    Vector3 offset = closest - s.center;

    if (offset.length2() < s.radius * s.radius) {
        // Collision
        printf("offset = %f %f %f\n", 
                offset.x, offset.y, offset.z);
        overlap = (s.radius - offset.length()) * 
            offset.normalize();
        return 1;
    } else {
        return 0;
    }
}

int intersectMovingOBBs2D(const OBB &a,
                          const OBB &b,
                          float &min_overlap,
                          Vector2 &min_overlap_axis)
{
    // Simply loop through the normals of a, then b to find a separating axis.
    // Very stupid.
    float minimum_overlap = FLT_MAX;
    Vector2 minimum_overlap_axis = {};

    auto find_sat = [&minimum_overlap, &minimum_overlap_axis]
                    (const OBB &a, const OBB &b, int idx) {
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

                b_min = MIN(b_dot_a_norm, b_min);
                b_max = MAX(b_dot_a_norm, b_max);
            }

            if (a_max < b_min || a_min > b_max) {
                // Found a separating axis
                return 1;
            }

            float overlap = 0.f;

            if (a_max >= b_min) {
                overlap = a_max - b_min;
            } else if (a_min >= b_max) {
                overlap = a_min - b_max;
            } else {
                assert(false);
            }

            if (overlap < minimum_overlap) {
                minimum_overlap = overlap;
                minimum_overlap_axis = (idx == 1) ? -1.f*a_norm : a_norm;
            }
        }

        return 0;
    };

    if (find_sat(a, b, 0)) {
        return 0;
    } else if (find_sat(b, a, 1)) {
        return 0;
    }

    min_overlap = minimum_overlap;
    min_overlap_axis = minimum_overlap_axis;

    return 1;
}

int intersectMovingOBBWall(const OBB &a,
                           const WallPlane &plane,
                           float &min_overlap)
{
    // Simply loop through the normals of a, then b to find a separating axis.
    // Very stupid.
    float minimum_overlap = FLT_MAX;

    for (int a_vert_idx = 0; a_vert_idx < 4; ++a_vert_idx) {
        Vector2 a_vert = a.verts[a_vert_idx];

        Vector2 diff = a_vert - plane.point;
        float dp = diff.dot(plane.normal);

        if (dp < minimum_overlap) {
            minimum_overlap = dp;
        }
    }

    min_overlap = minimum_overlap;

    if (minimum_overlap < 0.0f) {
        return 1;
    } else {
        return 0;
    }
}

int intersectMovingSphereWall(Sphere s,
                              Vector3 dx,
                              const WallPlane &plane,
                              float &t_out,
                              Vector3 &p_out)
{
    (void)p_out;

    Vector2 center_2d = { s.center.x, s.center.y };
    Vector2 normal_2d = { plane.normal.x, plane.normal.y };
    Vector2 dx_2d = { dx.x, dx.y };

    float t = (s.radius + plane.point.dot(normal_2d) - 
                          center_2d.dot(normal_2d)) /
              dx_2d.dot(plane.normal);

    if (t >= 0.0f && t <= 1.0f) {
        t_out = t * 0.9f;
        return 1;
    } else {
        return 0;
    }

#if 0
    float dist = plane.normal.dot({s.center.x, s.center.y}) -
                 plane.normal.dot(plane.point);

    if (std::abs(dist) <= s.radius) {
        t_out = 0.0f;
        p_out = s.center;
        return 1;
    } else {
        float denom = plane.normal.dot({dx.x, dx.y});
        if (denom * dist >= 0.f) {
            return 0;
        } else {
            float r = dist > 0.f ? s.radius : -s.radius;
            t_out = (r - dist) / denom;
            p_out = s.center + t_out * dx - 
                s.radius * Vector3{plane.normal.x, plane.normal.y, 0.f};

            if (t_out < 1.0f && t_out > 0.0f) {
                return 1;
            } else {
                return 0;
            }
        }
    }
#endif
}

int intersectSphereWall(Sphere s,
                        const WallPlane &plane,
                        float &min_overlap)
{
    Vector2 center_2d = { s.center.x, s.center.y };
    Vector2 normal_2d = { plane.normal.x, plane.normal.y };

    float dist = center_2d.dot(normal_2d) - plane.point.dot(normal_2d) -
                 s.radius;

    if (dist < 0.f) {
        min_overlap = -dist;
        return 1;
    } else {
        return 0;
    }
}

int intersectSphereWallSeg(Sphere s,
                           const WallSegment &seg,
                           float &min_overlap)
{
    Vector2 center_2d = { s.center.x, s.center.y };
    Vector2 normal_2d = { seg.normal.x, seg.normal.y };

    float dist = center_2d.dot(normal_2d) - seg.borders[0].dot(normal_2d) -
                 s.radius;

    if (dist < 0.f) {
        // If we are through the wall, check that we are within the bounds of
        // the boundaries
        Vector2 border_dir0 = seg.borders[1] - seg.borders[0];
        Vector2 border_dir1 = seg.borders[0] - seg.borders[1];

        Vector2 diff0 = center_2d - seg.borders[0];
        Vector2 diff1 = center_2d - seg.borders[1];

        if (diff0.dot(border_dir0) >= 0.f && diff1.dot(border_dir1) >= 0.f) {
            // There was a collision within the boundaries
            min_overlap = -dist;
            return 1;
        } else {
            return 0;
        }
    } else {
        return 0;
    }
}
    
}
