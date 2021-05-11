#pragma once
#include <nlohmann/json.hpp>

#include "Machines/layers/Conv1D.hpp"
#include "SerializeEigen.hpp"
#include "SerializeLayers.hpp"

CEREAL_CLASS_VERSION(yannq::Conv1D<float>, 1);
CEREAL_CLASS_VERSION(yannq::Conv1D<double>, 1);
CEREAL_CLASS_VERSION(yannq::Conv1D<long double>, 1);

namespace cereal
{
template<class Archive, typename T>
void save(Archive & ar, const yannq::Conv1D<T>& m, const uint32_t /*version*/)
{ 
	nlohmann::json desc = m.desc();
	ar(desc["use_bias"].get<bool>(),
			desc["input_channels"].get<int>(),
			desc["output_channels"].get<int>(), 
			desc["kernel_size"].get<int>(),
			desc["stride"].get<int>());
	typename yannq::Conv1D<T>::Vector pars = m.getParams();
	ar(pars);
}

template<class Archive, typename T>
void load(Archive& /*ar*/, yannq::Conv1D<T>& /*m*/, const uint32_t /*version*/)
{ 
	assert(false); // No default constructor or suitable modifier
}

template <typename T>
struct LoadAndConstruct<yannq::Conv1D<T> >
{
	template<class Archive>
	static void load_and_construct(Archive& ar,
			cereal::construct<yannq::Conv1D<T> >& construct,
			uint32_t const /*version*/)
	{
		bool useBias;
		int inChannels, outChannels, kernelSize, stride;
		ar(useBias, inChannels, outChannels, kernelSize, stride);

		construct(inChannels, outChannels, kernelSize, stride, useBias);
		typename yannq::Conv1D<T>::Vector pars;
		ar(pars);
		construct->setParams(pars);
	}
};
} //namespace cereal
LAYER_ADD_SERIALIZER(yannq::Conv1D, float);
LAYER_ADD_SERIALIZER(yannq::Conv1D, double);
LAYER_ADD_SERIALIZER(yannq::Conv1D, long double);
