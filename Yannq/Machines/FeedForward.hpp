#ifndef YANNQ_MACHINES_FEEDFORWARD_HPP
#define YANNQ_MACHINES_FEEDFORWARD_HPP
#include "Machines/layers/AbstractLayer.hpp"
#include "Utilities/Utility.hpp"
#include <iostream>
#include <sstream>
namespace yannq
{
template<typename T>
class FeedForward
{
public:
	using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
	using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
	using VectorRefType = Eigen::Ref<VectorType>;
	using VectorConstRefType = Eigen::Ref<const VectorType>;

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
	
	template<template<typename> class Layer, typename ...Ts>
	void addLayer(Ts&&... args)
	{
		auto layer = std::make_unique<Layer<T>>(args...);
		npar_ += layer->paramDim();
		layers_.push_back(std::move(layer));
	}

	VectorType getParams() const
	{
		uint32_t k = 0;
		VectorType par(npar_);
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

	void setParams(VectorConstRefType pars)
	{
		assert(pars.size() == npar_);
		uint32_t k = 0;
		for(auto& layer: layers_)
		{
			layer->setParams(pars.segment(k, layer->paramDim()));
			k += layer->paramDim();
		}
	}
	
	void clearLayers()
	{
		layers_.clear();
	}

	std::vector<VectorType> forward(const Eigen::VectorXi& sigma) const
	{
		using std::cosh;
		std::vector<VectorType> res;
		VectorType input = sigma.template cast<T>();
		res.push_back(input);
		for(auto& layer: layers_)
		{
			VectorType output(layer->outputDim(input.size()));
			layer->forward(input, output);
			res.push_back(output);
			input = std::move(output);
		}
		assert(input.size() == 1);
		return res;
	}

	VectorType backward(const std::vector<VectorType>& outs) const
	{
		assert(outs.size() == layers_.size() + 1);

		VectorType dw(npar_);
		VectorType dout(1);
		dout[0] = 1.;
		uint32_t k = npar_;
		for(int32_t idx = layers_.size() - 1; idx >= 0; --idx)
		{
			VectorType din(outs[idx].size());
			VectorType der(layers_[idx]->paramDim());
			layers_[idx]->backprop(outs[idx], outs[idx+1], dout, din, der);

			dout = std::move(din);
			dw.segment(k - layers_[idx]->paramDim(),layers_[idx]->paramDim()) = std::move(der);
			k -= layers_[idx]->paramDim();
		}
		return dw;
	}
};
}

#endif//YANNQ_MACHINES_FEEDFORWARD_HPP
