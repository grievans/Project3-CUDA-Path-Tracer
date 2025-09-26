#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<std::vector<KeyFrame>> geomFrames;

    void updateGeoms(float time);

    std::vector<Material> materials;
    float minT = 0.f;
    float maxT = 0.f;
    RenderState state;
};
