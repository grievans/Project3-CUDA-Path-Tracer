#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadFromGLTF(Geom& geom, const std::string& gltfName);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<std::vector<KeyFrame>> geomFrames;
    std::vector<glm::vec3> vertPositions;
    std::vector<glm::vec3> vertNormals;
    std::vector<Triangle> meshTriangles;

    void updateGeoms(float time);

    std::vector<Material> materials;
    float minT = 0.f;
    float maxT = 0.f;
    RenderState state;
};
