#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#include "TinyGLTF/tiny_gltf.h"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE,
    MESH
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct KeyFrame {
    float key;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
};


struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
    //virtual void update(float time);
    //std::vector<KeyFrame> frames;
    int triStart;
    int triEnd;
};


//void updateGeom(Geom* g, float time);

//struct AnimGeom : public Geom {
//    // TODO how do I want to store keyframes? I think want to recalculate matrices each time so can just interpolate vectors
//    // I think just a vec that's sorted by key should be fine? TODO not sure best way--map doesn't really work cuz wanna get to each key from values in between?
//    // maybe a std::set to make sure sorted? TODO
//    std::vector<KeyFrame> frames;
//    //void update(float time);
//};

struct Material
{
    glm::vec3 color;
    struct
    {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
    int lastMaterialID;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
};


struct Triangle {
    //glm::vec3 v0;
    //glm::vec3 v1;
    //glm::vec3 v2;
    int posIndices[3];
    int normIndices[3];
    
    // TODO materials and such once I do texture loading
    // TODO store normal indices in same as pos indices? since same values here
    glm::vec3 centroid; // TODO do I wanna store elsewhere since not needed I think on gpu side?

    // TODO need to store materials etc. within triangles rather than per mesh if wanting to do texturing later and also if wanting to facilitate loading multiple gltfs well for BVH
    // TODO maybe just storing positions directly actually is better? less steps of reading needed
    // TODO need to bake transformations into vert positions I think
};


struct BVHNode {
    glm::vec3 aabbMin, aabbMax;
    unsigned int leftNode;
    //bool isLeaf;
    unsigned int firstTriIdx, triCount;
    // TODO combine leftFirst
};