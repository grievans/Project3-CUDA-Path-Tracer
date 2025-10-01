#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}


// bringing over functions from 5610 since I've heard people say these versions are the cause of some precision issues I've had
// optimized algorithm for solving quadratic equations developed by Dr. Po-Shen Loh -> https://youtu.be/XKBX0r3J-9Y
// Adapted to root finding (ray t0/t1) for all quadric shapes (sphere, ellipsoid, cylinder, cone, etc.) by Erich Loftis
__host__ __device__ void solveQuadratic(float A, float B, float C, float& t0, float& t1) {
    float invA = 1.0 / A;
    B *= invA;
    C *= invA;
    float neg_halfB = -B * 0.5;
    float u2 = neg_halfB * neg_halfB - C;
    float u = u2 < 0.0 ? neg_halfB = 0.0 : sqrt(u2);
    t0 = neg_halfB - u;
    t1 = neg_halfB + u;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = 0.5f;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    


#if 0
    // TODO FIGURE OUT HOW MAKE WORK; still precision issues with other way it seems like?
    glm::vec3 rd = (multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));
    Ray ray;
    ray.origin = ro;
    ray.direction = rd;
    glm::vec3 pos = glm::vec3(0.f);

    //float sphereIntersect(Ray ray, float radius, vec3 pos, out vec3 localNor, out vec2 out_uv, mat4 invT) {
        //ray.origin = vec3(invT * vec4(ray.origin, 1.));
        //ray.direction = vec3(invT * vec4(ray.direction, 0.));
        float t0, t1;
        glm::vec3 diff = ray.origin - pos;
        float a = dot(ray.direction, ray.direction);
        float b = 2.0 * dot(ray.direction, diff);
        float c = dot(diff, diff) - (radius * radius);
        solveQuadratic(a, b, c, t0, t1);
        //normal = t0 > 0.0 ? ray.origin + t0 * ray.direction : ray.origin + t1 * ray.direction;
        normal = getPointOnRay(ray, t0 > 0.0 ? t0 : t1);
        //normal = normalize(normal);
        //normal = normalize(multiplyMV(sphere.))

        glm::vec3 objspaceIntersection = getPointOnRay(ray, t0 > 0.0 ? t0 : t1);

        // TODO just do r.origin + r.direction * t here? instead of converting space again
        //intersectionPoint = getPointOnRay(r, t0 > 0.0 ? t0 : t1);
        intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
        //normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
        normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(normal, 0.f)));

        //out_uv = sphereUVMap(normal);
        outside = glm::sign(t0) == glm::sign(t1);
        return t0 > 0.0 ? t0 : t1 > 0.0 ? t1 : -1.f;
    //}
#else
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));
    Ray rt;
    rt.origin = ro;
    rt.direction = rd;
    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
#if 0
    if (!outside)
    {
        normal = -normal; // TODO I've heard might want to remove depending on approach to refraction stuff
    }
#endif

    return glm::length(r.origin - intersectionPoint);
#endif
}


// Möller–Trumbore intersection
//#define TRIANGLE_EPSILON 0.0000001
//__host__ __device__ float triangleIntersect(const glm::vec3 &p0, const glm::vec3 &p1, const glm::vec3 &p2,
//    const glm::vec3 &rayOrigin, const glm::vec3 &rayDirection) {
//    
//    glm::vec3 edge1, edge2, h, s, q;
//    float a, f, u, v;
//    edge1 = p1 - p0;
//    edge2 = p2 - p0;
//    h = cross(rayDirection, edge2);
//    a = dot(edge1, h);
//    if (a > -TRIANGLE_EPSILON && a < TRIANGLE_EPSILON) {
//        return -1.f;    // This ray is parallel to this triangle.
//    }
//    f = 1.0 / a;
//    s = rayOrigin - p0;
//    u = f * dot(s, h);
//    if (u < 0.0 || u > 1.0)
//        return -1.f;
//    q = cross(s, edge1);
//    v = f * dot(rayDirection, q);
//    if (v < 0.0 || u + v > 1.0) {
//        return -1.f;
//    }
//    // At this stage we can compute t to find out where the intersection point is on the line.
//    float t = f * dot(edge2, q);
//    if (t > TRIANGLE_EPSILON) {
//        return t;
//    }
//    else // This means that there is a line intersection but not a ray intersection.
//        return -1.f;
//}


__host__ __device__ float triangleIntersectionTest(const Geom& mesh, const Triangle& tri, const glm::vec3 *vertPos, const glm::vec3 *vertNorm, Ray r, glm::vec3& intersectionPoint, glm::vec3& normal)
{
    // TODO maybe transform verts on CPU side initially before rather than transforming ray here?

    glm::vec3 ro = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = (multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    //rt.direction = (rd);
    rt.direction = rd;
    //rt.direction = glm::normalize(rd);

    const glm::vec3 & v0 = vertPos[tri.posIndices[0]];
    const glm::vec3 & v1 = vertPos[tri.posIndices[1]];
    const glm::vec3 & v2 = vertPos[tri.posIndices[2]];

    //triangleIntersect(v0, v1, v2, rt.origin, rt.direction);
    

    glm::vec3 bPos;
    // TODO intersection still wrong it seems like
    if (glm::intersectRayTriangle(rt.origin, rt.direction, v0, v1, v2, bPos)) {

    }
    else if (glm::intersectRayTriangle(rt.origin, rt.direction, v0, v2, v1, bPos)) {
        //std::swap(bPos.x, bPos.y);
        float tmp = bPos.x;
        bPos.x = bPos.y;
        bPos.y = tmp;
    } else {
    //if (!(glm::intersectRayTriangle(rt.origin, rt.direction, v0, v1, v2, bPos))) {
    //if (!(glm::intersectRayTriangle(rt.origin, rt.direction, v0, v2, v1, bPos))) {
    //if (!(glm::intersectRayTriangle(rt.origin, rt.direction, v0, v1, v2, bPos) || glm::intersectRayTriangle(rt.origin, rt.direction, v0, v2, v1, bPos))) {
        return -1.f;
    }
    const glm::vec3 & n0 = vertNorm[tri.normIndices[0]];
    const glm::vec3 & n1 = vertNorm[tri.normIndices[1]];
    const glm::vec3 & n2 = vertNorm[tri.normIndices[2]];

    // TODO need to figure out how to do this
    // also maybe need to 
    //std::swap(bPos.x, bPos.y);
    //printf("%f == %f %f\n", bPos.z, 1.f - bPos.x - bPos.y, bPos.z == 1.f - bPos.x - bPos.y);
    //bPos.z = 1.f - bPos.x - bPos.y;
    float w = 1.f - bPos.x - bPos.y;
    //printf("%f %f %f %f\n", bPos.z, bPos.x, bPos.y, bPos.x + bPos.y + bPos.z);
    //glm::vec3 iPos = (w * v0 + bPos.x * v1 + bPos.y * v2);
    //float localDist = glm::length(rt.origin - iPos);
    //glm::vec3 iPos = (bPos.x * v0 + bPos.y * v1 + bPos.z * v2);
    //iPos = multiplyMV(mesh.transform, glm::vec4(iPos,1.f));
    //intersectionPoint = iPos;

    intersectionPoint = r.origin + r.direction * bPos.z;
    //normal = (bPos.z * n0 + bPos.x * n1 + bPos.y * n2);
    //normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(n0, 0.f)));
    normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4((w * n0 + bPos.x * n1 + bPos.y * n2), 0.f)));
    //return 0.2f;
    return glm::length(r.origin - intersectionPoint);
}
