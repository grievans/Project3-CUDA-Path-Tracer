#pragma once

#include "sceneStructs.h"
#include <vector>


#define USE_BVH 0

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadFromGLTF(Geom& geom, const std::string& gltfName);

    void updateNodeBounds(unsigned int nodeIdx);
    void subdivide(unsigned int nodeIdx);

    void transformTriangles();
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<Geom> meshGeoms;
    // geoms is passed to GPU, meshGeoms only used for tracking what transformations apply to what triangles

    std::vector<std::vector<KeyFrame>> geomFrames;
    std::vector<std::vector<KeyFrame>> meshGeomFrames;
    std::vector<glm::vec3> originalVertPositions;
    std::vector<glm::vec3> vertPositions;
    std::vector<glm::vec3> originalVertNormals;
    std::vector<glm::vec3> vertNormals;
    std::vector<Triangle> meshTriangles;
    std::vector<BVHNode> bvhNode;
    std::vector<unsigned int> triIdx;

    void updateGeoms(float time);

    void buildBVH();

    std::vector<Material> materials;
    float minT = 0.f;
    float maxT = 0.f;
    RenderState state;

    unsigned int nodesUsed;
};
