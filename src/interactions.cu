#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}


#define PDF_EPSILON 0.00001f
__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.



    // TODO figure out how I want to structure the different states

    /*if (pathSegment.remainingBounces <= 0) {
        pathSegment.color = glm::vec3(0.f);
        return;
    }*/

    if (m.emittance > 0.0f) {
        //pathSegment.color = glm::vec3(1000.f);
        pathSegment.color *= (m.color * m.emittance);
        pathSegment.remainingBounces = 0;
        return;
    }

    // TODO need last bounce to also become 0 color I guess?
    if (pathSegment.remainingBounces < 2) { // <= 1; no chance to hit light at this point so no reason to bounce again
        pathSegment.color = glm::vec3(0.f);
        pathSegment.remainingBounces = 0;
        return;
    }

    // TODO specular intensity and diffuse intensity, store in material perhaps? or just sum here. then rng choose based on weights and render that one
    //   potential thing: should randomly choose in prior iteration and sort based on that?
    // TODO not sure if intensity should be some other measure--just length?
    float diffIntensity = m.color.r + m.color.g + m.color.b;
    float specIntensity = m.specular.color.r + m.specular.color.g + m.specular.color.b;
    thrust::uniform_real_distribution<float> u01(0, 1);
    float randWeight = u01(rng) * (diffIntensity + specIntensity);

    if (randWeight < specIntensity) { //TODO 
        pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
            //TODO imperfect specular reflection
        float absDotDirNor = abs(dot(pathSegment.ray.direction, normal));

        pathSegment.color *= m.specular.color * absDotDirNor; // / pdf -> / 1

        --pathSegment.remainingBounces;
        return;
    }


    // Super basic diffuse
    pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
    float absDotDirNor = abs(dot(pathSegment.ray.direction, normal));
    // TODO should ray.direction store in a separate variable and use that in ^ rather than directly putting .direction again? unsure of affect on memory access
    float pdf = absDotDirNor * INV_PI; // TODO verify this is right, also should possibly do * INV_PI with another constant storing that instead of dividing?
    // oh wait is this not cosine weighted? seems wrong when I divide....or wait is this the one that just works out?
    if (pdf < PDF_EPSILON) { // TODO < epsilon? still not sure how I want to do
        pathSegment.color = glm::vec3(0.f);
        pathSegment.remainingBounces = 0;
        return;
    }
    pathSegment.ray.origin = intersect;


    pathSegment.color *= m.color * INV_PI * absDotDirNor / pdf; // TODO does that need a scale?
    --pathSegment.remainingBounces;
    //--pathSegment.remainingBounces;

}
