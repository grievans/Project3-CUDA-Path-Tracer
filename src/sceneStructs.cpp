#include "sceneStructs.h"
#include <glm/gtc/matrix_inverse.hpp>
#include "utilities.h"

//void updateGeom(Geom* g, float time) {
//    //AnimGeom* ag = dynamic_cast<AnimGeom*>(g);
//
//    if (g->frames.size() > 0) {
//        KeyFrame* f0 = &(g->frames[0]);
//        KeyFrame* f1 = &(g->frames[0]);
//        for (unsigned int i = 0; i < g->frames.size(); ++i) {
//            if (g->frames[i].key >= time) {
//                f0 = &(g->frames[i - 1]);
//                f1 = &(g->frames[i]);
//                break;
//            }
//        }
//        float u = (time - f0->key) / (f1->key - f0->key);
//        g->translation = glm::mix(f0->translation, f1->translation, u);
//        // TODO SLERP?
//        g->rotation = glm::mix(f0->rotation, f1->rotation, u);
//        g->scale = glm::mix(f0->scale, f1->scale, u);
//
//
//        g->transform = utilityCore::buildTransformationMatrix(
//            g->translation, g->rotation, g->scale);
//        g->inverseTransform = glm::inverse(g->transform);
//        g->invTranspose = glm::inverseTranspose(g->transform);
//    }
//    return;
//}

//void Geom::update(float time) {
//    return; // Do nothing
//}
//
//void AnimGeom::update(float time)
//{
//    // TODO should this be GPU-side? probably won't have enough objects to make worth it
//    KeyFrame* f0 = &frames[0];
//    KeyFrame* f1 = &frames[0];
//    for (unsigned int i = 0; i < frames.size(); ++i) {
//        if (frames[i].key >= time) {
//            f0 = &frames[i - 1];
//            f1 = &frames[i];
//            break;
//        }
//    }
//    float u = (time - f0->key) / (f1->key - f0->key);
//    this->translation = glm::mix(f0->translation, f1->translation, u);
//    // TODO SLERP?
//    this->rotation = glm::mix(f0->rotation, f1->rotation, u);
//    this->scale = glm::mix(f0->scale, f1->scale, u);
//
//    
//    this->transform = utilityCore::buildTransformationMatrix(
//        this->translation, this->rotation, this->scale);
//    this->inverseTransform = glm::inverse(this->transform);
//    this->invTranspose = glm::inverseTranspose(this->transform);
//    return;
//
//}
