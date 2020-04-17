#pragma once

#include <cereal/types/polymorphic.hpp>

#include "Machines/layers/ActivationLayer.hpp"
#include "SerializeLayers.hpp"

/**
 * TODO: May use a fancy macro for simplification.
 * Somehow does not work...?
 */


LAYER_ADD_SERIALIZER(yannq::Identity, float);
LAYER_ADD_SERIALIZER(yannq::Identity, double);
LAYER_ADD_SERIALIZER(yannq::Identity, long double);

LAYER_ADD_SERIALIZER(yannq::LnCosh, float);
LAYER_ADD_SERIALIZER(yannq::LnCosh, double);
LAYER_ADD_SERIALIZER(yannq::LnCosh, long double);

LAYER_ADD_SERIALIZER(yannq::Tanh, float);
LAYER_ADD_SERIALIZER(yannq::Tanh, double);
LAYER_ADD_SERIALIZER(yannq::Tanh, long double);

LAYER_ADD_SERIALIZER(yannq::Sigmoid, float);
LAYER_ADD_SERIALIZER(yannq::Sigmoid, double);
LAYER_ADD_SERIALIZER(yannq::Sigmoid, long double);

LAYER_ADD_SERIALIZER(yannq::ReLU, float);
LAYER_ADD_SERIALIZER(yannq::ReLU, double);
LAYER_ADD_SERIALIZER(yannq::ReLU, long double);

LAYER_ADD_SERIALIZER(yannq::LeakyReLU, float);
LAYER_ADD_SERIALIZER(yannq::LeakyReLU, double);
LAYER_ADD_SERIALIZER(yannq::LeakyReLU, long double);

LAYER_ADD_SERIALIZER(yannq::HardTanh, float);
LAYER_ADD_SERIALIZER(yannq::HardTanh, double);
LAYER_ADD_SERIALIZER(yannq::HardTanh, long double);

LAYER_ADD_SERIALIZER(yannq::SoftShrink, float);
LAYER_ADD_SERIALIZER(yannq::SoftShrink, double);
LAYER_ADD_SERIALIZER(yannq::SoftShrink, long double);

LAYER_ADD_SERIALIZER(yannq::LeakyHardTanh, float);
LAYER_ADD_SERIALIZER(yannq::LeakyHardTanh, double);
LAYER_ADD_SERIALIZER(yannq::LeakyHardTanh, long double);

LAYER_ADD_SERIALIZER(yannq::SoftSign, float);
LAYER_ADD_SERIALIZER(yannq::SoftSign, double);
LAYER_ADD_SERIALIZER(yannq::SoftSign, long double);

//CEREAL_FORCE_DYNAMIC_INIT(YANNQ);

