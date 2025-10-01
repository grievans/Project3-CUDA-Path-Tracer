#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::updateGeoms(float time)
{
    for (int index = 0; index < this->geoms.size(); ++index) {
        Geom* g = &(this->geoms.at(index));
        std::vector<KeyFrame>& frames = this->geomFrames.at(index);
        //AnimGeom* ag = dynamic_cast<AnimGeom*>(g);

        if (frames.size() > 1) {
            KeyFrame* f0 = &(frames[0]);
            KeyFrame* f1 = &(frames[1]);
            for (unsigned int i = 1; i < frames.size(); ++i) {
                if (frames[i].key >= time) {
                    f0 = &(frames[i - 1]);
                    f1 = &(frames[i]);
                    break;
                }
            }
            float u = (time - f0->key) / (f1->key - f0->key);
            g->translation = glm::mix(f0->translation, f1->translation, u);
            // TODO SLERP?
            g->rotation = glm::mix(f0->rotation, f1->rotation, u);
            g->scale = glm::mix(f0->scale, f1->scale, u);


            g->transform = utilityCore::buildTransformationMatrix(
                g->translation, g->rotation, g->scale);
            g->inverseTransform = glm::inverse(g->transform);
            g->invTranspose = glm::inverseTranspose(g->transform);
        }
    }
    return;
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specular.color = glm::vec3(0.f);
            newMaterial.hasReflective = 1.f;
            newMaterial.hasRefractive = 0.f;
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.specular.color = glm::vec3(0.f);
            newMaterial.emittance = p["EMITTANCE"];
            newMaterial.hasReflective = 1.f;
            newMaterial.hasRefractive = 0.f;
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(0.f);
            newMaterial.specular.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 1.f;
            newMaterial.hasRefractive = 0.f;
        }
        else if (p["TYPE"] == "Specular Transmissive") {
            const auto& col = p["RGB"];
            newMaterial.specular.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.color = glm::vec3(0.f);
            newMaterial.hasReflective = 0.f;
            newMaterial.hasRefractive = 1.f;
            newMaterial.indexOfRefraction = p["ETA"]; 
        }
        else if (p["TYPE"] == "Glass") {
            const auto& col = p["RGB"];
            newMaterial.specular.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.color = glm::vec3(0.f);
            newMaterial.hasReflective = 1.f;
            newMaterial.hasRefractive = 1.f;
            newMaterial.indexOfRefraction = p["ETA"]; 
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        
        Geom newGeom;
        std::vector<KeyFrame> newFrames;
        if (p.contains("FRAMES")) {
            const auto& animData = p["FRAMES"];

            // TODO how do I wanna store animation?
            // TODO should I do some sort of slerp to allow rotation keyframing?

            for (const auto& f : animData) {
                float t = f["T"];
                const auto& trans = f["TRANS"];
                const auto& rotat = f["ROTAT"];
                const auto& scale = f["SCALE"];
                KeyFrame frame;
                frame.translation = glm::vec3(trans[0], trans[1], trans[2]);
                frame.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
                frame.scale = glm::vec3(scale[0], scale[1], scale[2]);
                frame.key = t;
                // TODO note this only supports time scales which include 0 as a point rn
                if (t > this->maxT) {
                    this->maxT = t;
                }
                if (t < this->minT) {
                    this->minT = t;
                }
                newFrames.push_back(frame);
                

            }
            if (type == "cube")
            {
                newGeom.type = CUBE;
            }
            else
            {
                newGeom.type = SPHERE;
            }
            newGeom.materialid = MatNameToID[p["MATERIAL"]];

            newGeom.translation = newFrames[0].translation;
            newGeom.rotation = newFrames[0].rotation;
            newGeom.scale = newFrames[0].scale;
            newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
            newGeom.inverseTransform = glm::inverse(newGeom.transform);
            newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

            //geoms.push_back(newGeom);
        }
        else {
            if (type == "cube")
            {
                newGeom.type = CUBE;
            }
            else if (type == "gltf") {
                newGeom.type = MESH;
                const auto& file = p["FILE"];
                loadFromGLTF(newGeom, file);
            }
            else
            {
                newGeom.type = SPHERE;
            }
            newGeom.materialid = MatNameToID[p["MATERIAL"]];
            const auto& trans = p["TRANS"];
            const auto& rotat = p["ROTAT"];
            const auto& scale = p["SCALE"];
            newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
            newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
            newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
            newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
            newGeom.inverseTransform = glm::inverse(newGeom.transform);
            newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);


        }
        geoms.push_back(newGeom);
        geomFrames.push_back(newFrames);
        

    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

void Scene::loadFromGLTF(Geom& geom, const std::string& gltfName)
{
    //following examples/basic/main in tinygltf
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    tinygltf::Model model;

    // TODO read file path to check if binary vs. ASCII?
    bool res = loader.LoadASCIIFromFile(&model, &err, &warn, gltfName);
    //bool res = loader.LoadBinaryFromFile(&model, &err, &warn, gltfName);
    if (!warn.empty()) {
        std::cout << "WARN: " << warn << std::endl;
    }

    if (!err.empty()) {
        std::cout << "ERR: " << err << std::endl;
    }

    if (!res)
        std::cout << "Failed to load glTF: " << gltfName << std::endl;
    else
        std::cout << "Loaded glTF: " << gltfName << std::endl;


    geom.triStart = this->meshTriangles.size();

    // TODO

    for (const auto& mesh : model.meshes) {
        std::cout << "Loading mesh: " << mesh.name << std::endl;
        for (const auto& prim : mesh.primitives) {
            
            tinygltf::Accessor& posAccessor = model.accessors[prim.attributes.at("POSITION")];
            tinygltf::BufferView& posBufferView = model.bufferViews[posAccessor.bufferView];
            tinygltf::Buffer& posBuffer = model.buffers[posBufferView.buffer];
            //tinygltf::Accessor& normAccessor = model.accessors[prim.attributes.at("NORMAL")];
            // TODO figure out syntax for loading in the points from this
            // TODO checkout perhaps accessor.minVal for later
            //const auto posDataPtr = posBuffer.data.data() + posBufferView.byteOffset + posAccessor.byteOffset;
            const float* posDataPtr = (float*) &(posBuffer.data[posBufferView.byteOffset + posAccessor.byteOffset]);
            //const auto posByteStride = posAccessor.ByteStride(posBufferView);
            const auto posCount = posAccessor.count;

            int vertStartidx = this->vertPositions.size();

            for (size_t i = 0; i < posCount; ++i) {
                glm::vec3 pos(posDataPtr[i * 3], posDataPtr[i * 3 + 1], posDataPtr[i * 3 + 2]);
                //pos *= 100.f;
                this->vertPositions.push_back(pos);
            }


            tinygltf::Accessor& normAccessor = model.accessors[prim.attributes.at("NORMAL")];
            tinygltf::BufferView& normBufferView = model.bufferViews[normAccessor.bufferView];
            tinygltf::Buffer& normBuffer = model.buffers[normBufferView.buffer];

            //const auto normDataPtr = normBuffer.data.data() + normBufferView.byteOffset + normAccessor.byteOffset;
            const float* normDataPtr = (float*)&(normBuffer.data[normBufferView.byteOffset + normAccessor.byteOffset]);
            const auto normCount = normAccessor.count;
            // TODO make support no normal data
            for (size_t i = 0; i < normCount; ++i) {
                glm::vec3 norm(normDataPtr[i * 3], normDataPtr[i * 3 + 1], normDataPtr[i * 3 + 2]);
                this->vertNormals.push_back(norm);
            }
                
            tinygltf::Accessor& indexAccessor = model.accessors[prim.indices];
            tinygltf::BufferView& indexBufferView = model.bufferViews[indexAccessor.bufferView];
            tinygltf::Buffer& indexBuffer = model.buffers[indexBufferView.buffer];
            
            //const auto indexDataPtr = indexBuffer.data.data() + indexBufferView.byteOffset + indexAccessor.byteOffset;
            // TODO have to do something to make types agnostic
            const uint16_t* indexDataPtr = (uint16_t*)&(indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset]);
            const auto indexCount = indexAccessor.count;
            
            
            
            for (size_t i = 0; i < indexCount; i += 3) {
                Triangle tri;
                for (int j = 0; j < 3; ++j) {

                    tri.posIndices[j] = vertStartidx + static_cast<int>(indexDataPtr[i + j]);
                    tri.normIndices[j] = vertStartidx + static_cast<int>(indexDataPtr[i + j]);

                }
                this->meshTriangles.push_back(tri);
            }
        }
    }

    // placeholder values to make sure triangle intersection working
   /* Triangle tri;
    for (int i = 0; i < 3; ++i) {
        tri.posIndices[i] = i;
        tri.normIndices[i] = i;

    }
    this->vertPositions.push_back(glm::vec3(1.f, 1.f, 0.f));
    this->vertPositions.push_back(glm::vec3(-1.f, -1.f, 0.f));
    this->vertPositions.push_back(glm::vec3(1.f, -1.f, 0.f));
    this->vertNormals.push_back(glm::vec3(0.f, 0.f, -1.f));
    this->vertNormals.push_back(glm::vec3(0.f, 0.f, -1.f));
    this->vertNormals.push_back(glm::vec3(0.f, 0.f, -1.f));
    this->meshTriangles.push_back(tri);*/

    geom.triEnd = this->meshTriangles.size();

}
