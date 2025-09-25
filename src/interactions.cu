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


__host__ __device__ float fresnelDielectricEval(float cosThetaI) {
    // TODO add
    // TODO probably needs m.indexOfRefraction

    float Rparl = 1.f;
    float Rperp = 1.f;
    return (Rparl * Rparl + Rperp * Rperp) / 2.f;
}

__host__ __device__ bool refract(glm::vec3 wi, glm::vec3 n, float eta, glm::vec3& wt) {
    float cosThetaI = dot(n, wi);
    float sin2ThetaI = glm::max(0.f, 1.f - cosThetaI * cosThetaI);
    float sin2ThetaT = eta * eta * sin2ThetaI;

    if (sin2ThetaT >= 1.f) return false; // TODO might change how these values work
    float cosThetaT = sqrt(1.f - sin2ThetaT);
    wt = eta * -wi + (eta * cosThetaI - cosThetaT) * n;
    return true;

}

#define PDF_EPSILON 0.0001f
#define RAY_EPSILON 0.01f
// TODO figure out good epsilons; IDK why I have issues for 0.001f step along distance
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

    thrust::uniform_real_distribution<float> u01(0, 1);
    
    float modeSum = m.hasReflective + m.hasRefractive;
    float refMode = u01(rng) * (modeSum); // TODO make better variable names?
    // if has both reflective and refractive components, 50% chance of each and modeSum = 2.f -> multiply color by 2
    // otherwise just has the one
    // TODO precomputing random values in previous step and including that in sorting seems like it'd be beneficial?

    float diffIntensity = m.color.r + m.color.g + m.color.b;
    float specIntensity = m.specular.color.r + m.specular.color.g + m.specular.color.b;

    float randWeight = u01(rng) * (diffIntensity + specIntensity);




    // TODO specular intensity and diffuse intensity, store in material perhaps? or just sum here. then rng choose based on weights and render that one
    //   potential thing: should randomly choose in prior iteration and sort based on that?
    // TODO not sure if intensity should be some other measure--just length?
    //pathSegment.ray.origin = intersect;

    if (randWeight < specIntensity) { //TODO 

        if (refMode < m.hasRefractive) {
            //refraction:
            
            // TOOD going to write this code separately first then look into how can combine

            //pathSegment.ray.direction = normalize(pathSegment.ray.direction);


            bool entering = dot(normal, pathSegment.ray.direction) < 0;
            float eta = entering ? 1.f / m.indexOfRefraction : m.indexOfRefraction;
            //float eta = entering ? m.indexOfRefraction : 1.f / m.indexOfRefraction;

            glm::vec3 wi = glm::refract(pathSegment.ray.direction, entering ? normal : -normal, eta);
            //if (length(wi) < 0.000001f) {
            // TODO should this be == 0 or < epsilon?
            if (wi == glm::vec3(0.f)) {
            //glm::vec3 wi;
            //if (!refract(pathSegment.ray.direction, entering ? normal : -normal, eta, wi)) {
                pathSegment.color = glm::vec3(0.f);
                pathSegment.remainingBounces = 0;
                return;
            }
            pathSegment.ray.direction = normalize(wi);
            // TODO normalize stuff?
            //pathSegment.ray.direction = normalize(wi);
            //pathSegment.ray.direction = normalize(pathSegment.ray.direction);

            pathSegment.ray.origin = intersect + pathSegment.ray.direction * RAY_EPSILON;
            //float absDotDirNor = abs(dot(pathSegment.ray.direction, normal)); //cancels out
            pathSegment.color *= m.specular.color * eta * eta * modeSum;
            --pathSegment.remainingBounces;
            return;
        }

        //reflection:

        pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.ray.origin = intersect + pathSegment.ray.direction * RAY_EPSILON;
            //TODO imperfect specular reflection
        //float absDotDirNor = abs(dot(pathSegment.ray.direction, normal)); //cancels out here

        //if (m.hasRefractive) {
            //pathSegment.ray.direction *= -1.f; //TODO not how to do this just test
        //}

        pathSegment.color *= m.specular.color * modeSum; // / pdf -> / 1
        
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
    pathSegment.ray.origin = intersect + pathSegment.ray.direction * RAY_EPSILON;


    pathSegment.color *= m.color * INV_PI * absDotDirNor / pdf * modeSum; // TODO does that need a scale?
    --pathSegment.remainingBounces;
    //--pathSegment.remainingBounces;

}
