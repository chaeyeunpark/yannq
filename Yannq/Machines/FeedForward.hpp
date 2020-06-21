#ifndef YANNQ_MACHINES_FEEDFORWARD_HPP
#define YANNQ_MACHINES_FEEDFORWARD_HPP
#include <iostream>
#include <sstream>

#include "layers/AbstractLayer.hpp"
#include "Utilities/Utility.hpp"
namespace yannq
{
enum class InitializationMode
{
	LeCun, Xavier, He
};
template<typename T>
class FeedForward
{
public:
	using Scalar = T;
	using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
	using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
	using VectorRef = Eigen::Ref<Vector>;
	using VectorConstRef = Eigen::Ref<const Vector>;

private:
	std::vector<std::unique_ptr<AbstractLayer<T>>> layers_;

	uint32_t npar_ = 0;

public:
	explicit FeedForward()
	{
	}

	template<typename RandomEngine>
	void initializeRandom(RandomEngine& re, typename remove_complex<T>::type sigma)
	{
		auto randVec = randomVector<T>(std::forward<RandomEngine>(re), sigma, npar_);
		setParams(randVec);
	}

	template<typename RandomEngine>
	void initializeRandom(RandomEngine& re, InitializationMode mode)
	{
		using std::sqrt;

		auto sigma = [mode](uint32_t fanIn, uint32_t fanOut)
		{
			switch(mode)
			{
			case InitializationMode::LeCun:
				return sqrt(1.0/fanIn);
			case InitializationMode::Xavier:
				return sqrt(2.0/(fanIn + fanOut));
			case InitializationMode::He:
				return sqrt(2.0/(fanIn));
			}
			assert(false);
			return 0.0;
		};

		for(auto& layer: layers_)
		{
			if(layer->paramDim() == 0)
				continue;
			layer->setParams(
				randomVector<T>(
					std::forward<RandomEngine>(re),
					sigma(layer->fanIn(), layer->fanOut()),
					layer->paramDim())
			);
		}
	}

	
	template<template<typename> class Layer, typename ...Ts>
	void addLayer(Ts&&... args)
	{
		auto layer = std::make_unique<Layer<T>>(args...);
		npar_ += layer->paramDim();
		layers_.push_back(std::move(layer));
	}

	uint32_t getDim() const
	{
		return npar_;
	}

	Vector getParams() const
	{
		uint32_t k = 0;
		Vector par(npar_);
		for(auto& layer: layers_)
		{
			par.segment(k, layer->paramDim()) = layer->getParams();
			k += layer->paramDim();
		}
		return par;
	}

	std::string summary() const
	{
		std::ostringstream ss;
		ss << "Layer name\tNumber of params" << std::endl;
		for(const auto& layer: layers_)
		{
			ss << layer->name() << "\t" << layer->paramDim() << std::endl;
		}
		return ss.str();
	}

	nlohmann::json desc() const
	{
		nlohmann::json res;
		for(const auto& layer: layers_)
		{
			res.push_back( layer->desc() );
		}
		return res;
	}

	void setParams(VectorConstRef pars)
	{
		assert(pars.size() == npar_);
		uint32_t k = 0;
		for(auto& layer: layers_)
		{
			layer->setParams(pars.segment(k, layer->paramDim()));
			k += layer->paramDim();
		}
	}

	void updateParams(VectorConstRef ups)
	{
		assert(ups.size() == npar_);
		uint32_t k = 0;
		for(auto& layer: layers_)
		{
			if(layer->paramDim() == 0)
				continue;

			layer->updateParams(ups.segment(k, layer->paramDim()));
			k += layer->paramDim();
		}
	}
	
	void clearLayers()
	{
		layers_.clear();
	}

	Scalar forward(const Eigen::VectorXi& sigma) const
	{
		Vector input = sigma.template cast<T>();
		for(auto& layer: layers_)
		{
			Vector output(layer->outputDim(input.size()));
			layer->forward(input, output);
			input = std::move(output);
		}
		assert(input.size() == 1);
		return input(0);
	}

	Scalar forward(const std::vector<Vector>& data) const
	{
		return data.back()(0);
	}

	std::vector<Vector> makeData(const Eigen::VectorXi& sigma) const
	{
		std::vector<Vector> res;
		Vector input = sigma.template cast<T>();
		res.push_back(input);
		for(auto& layer: layers_)
		{
			Vector output(layer->outputDim(input.size()));
			layer->forward(input, output);
			res.push_back(output);
			input = std::move(output);
		}
		assert(input.size() == 1);
		return res;
	}

	Vector backward(const std::vector<Vector>& data) const
	{
		assert(data.size() == layers_.size() + 1);

		Vector dw(npar_);
		Vector dout(1);
		dout[0] = 1.;
		uint32_t k = npar_;
		for(int32_t idx = layers_.size() - 1; idx >= 0; --idx)
		{
			Vector din(data[idx].size());
			Vector der(layers_[idx]->paramDim());
			layers_[idx]->backprop(data[idx], data[idx+1], dout, din, der);

			dout = std::move(din);
			dw.segment(k - layers_[idx]->paramDim(),layers_[idx]->paramDim()) = std::move(der);
			k -= layers_[idx]->paramDim();
		}
		return dw;
	}
};
}

#endif//YANNQ_MACHINES_FEEDFORWARD_HPP
