#pragma once
#include <nlohmann/json.hpp>

#include "Machines/layers/FullyConnected.hpp"
#include "SerializeEigen.hpp"
#include "SerializeLayers.hpp"

CEREAL_CLASS_VERSION(yannq::FullyConnected<float>, 1);
CEREAL_CLASS_VERSION(yannq::FullyConnected<double>, 1);
CEREAL_CLASS_VERSION(yannq::FullyConnected<long double>, 1);

namespace cereal
{
template<class Archive, typename T>
void save(Archive & ar, const yannq::FullyConnected<T>& m, const uint32_t /*version*/)
{ 
	nlohmann::json desc = m.desc();
	ar(desc["use_bias"].get<bool>(),
			desc["input_dim"].get<int>(),
			desc["output_dim"].get<int>());
	typename yannq::FullyConnected<T>::Vector pars = m.getParams();
	ar(pars);
}

template<class Archive, typename T>
void load(Archive & ar, yannq::FullyConnected<T>& m, const uint32_t /*version*/)
{ 
	assert(false); // No default constructor or suitable modifier
}

template <typename T>
struct LoadAndConstruct<yannq::FullyConnected<T> >
{
	template<class Archive>
	static void load_and_construct(Archive& ar,
			cereal::construct<yannq::FullyConnected<T> >& construct,
			uint32_t const /*version*/)
	{
		bool useBias;
		int inputDim, outputDim;
		ar(useBias, inputDim, outputDim);

		construct(inputDim, outputDim, useBias);
		typename yannq::FullyConnected<T>::Vector pars;
		ar(pars);
		construct->setParams(pars);
	}
};
} //namespace cereal
LAYER_ADD_SERIALIZER(yannq::FullyConnected, float);
LAYER_ADD_SERIALIZER(yannq::FullyConnected, double);
LAYER_ADD_SERIALIZER(yannq::FullyConnected, long double);
