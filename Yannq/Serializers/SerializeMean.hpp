#pragma once
#include <nlohmann/json.hpp>

#include "Machines/layers/Mean.hpp"
#include "SerializeEigen.hpp"
#include "SerializeLayers.hpp"

CEREAL_CLASS_VERSION(yannq::Mean<float>, 1);
CEREAL_CLASS_VERSION(yannq::Mean<double>, 1);
CEREAL_CLASS_VERSION(yannq::Mean<long double>, 1);

namespace cereal
{
template<class Archive, typename T>
void save(Archive & ar, const yannq::Mean<T>& m, const uint32_t /*version*/)
{ 
	nlohmann::json desc = m.desc();
	auto inputDim = desc["input_dim"].get<std::array<uint32_t, 2>>();
	ar(inputDim[0], inputDim[1], desc["mean_axis"].get<uint32_t>());
}

template<class Archive, typename T>
void load(Archive& /*ar*/, yannq::Mean<T>& /*m*/, const uint32_t /*version*/)
{ 
	assert(false); // No default constructor or suitable modifier
}

template <typename T>
struct LoadAndConstruct<yannq::Mean<T> >
{
	template<class Archive>
	static void load_and_construct(Archive& ar, 
			cereal::construct<yannq::Mean<T> >& construct,
			uint32_t const /*version*/)
	{
		std::array<uint32_t, 2> inputDim;
		uint32_t meanAxis;
		ar(inputDim[0], inputDim[1], meanAxis);

		construct(inputDim, meanAxis);
	}
};
} //namespace cereal

LAYER_ADD_SERIALIZER(yannq::Mean, float);
LAYER_ADD_SERIALIZER(yannq::Mean, double);
LAYER_ADD_SERIALIZER(yannq::Mean, long double);
