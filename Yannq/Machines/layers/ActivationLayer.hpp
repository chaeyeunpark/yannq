// Modified from NetKet source for YANNQ Project
// <Chae-Yeun Park>(chae.yeun.park@gmail.com)
// Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef YANNQ_ACTIVATIONLAYER_HH
#define YANNQ_ACTIVATIONLAYER_HH

#include <Eigen/Dense>
#include <complex>
#include <fstream>
#include <random>
#include <vector>
#include "AbstractLayer.hpp"
#include "activations.hpp"

namespace yannq {



template <typename T, class Activation>
class ActivationLayer : public AbstractLayer<T> {
	using VectorType = typename AbstractLayer<T>::VectorType;
	using MatrixType = typename AbstractLayer<T>::MatrixType;
	using VectorRefType = typename AbstractLayer<T>::VectorRefType;
	using VectorConstRefType = typename AbstractLayer<T>::VectorConstRefType;

	Activation activation_;  // activation

	static constexpr char name_[] = "Activation Layer";

public:
	template<typename ...Ts>
	ActivationLayer(Ts&&... args)
		: activation_(std::forward<Ts>(args)...)
	{
	}

	std::string name() const override { return name_; }

	uint32_t paramDim() const { return 0; }

	VectorType getParams() const
	{
		return VectorType{};
	}

	void setParams(VectorConstRefType pars)
	{
		(void)pars;
	}

	uint32_t outputDim(uint32_t inputDim) const override {
		return inputDim; 
	}

	// Feedforward
	void forward(const VectorType &input, VectorType &output) override 
	{
		activation_->operator()(input, output);
	}

	// Computes derivative.
	void backprop(const VectorType &prev_layer_output,
			const VectorType &this_layer_output, const VectorType &dout,
			VectorType &din, VectorRefType /*der*/) override 
	{
		din.resize(prev_layer_output.size());
		activation_->ApplyJacobian(prev_layer_output, this_layer_output, dout, din);
	}

	nlohmann::json to_json() const override
	{
		nlohmann::json layerpar;
		layerpar["name"] = name_;
		layerpar["activation"] = activation_->to_json();
		return layerpar;
	}
};
template <typename T, class Activation>
constexpr char ActivationLayer<T, Activation>::name_[];

template<typename T>
using LayerReLU = ActivationLayer<T, ReLU<T> >;

template<typename T>
using LayerLnCosh = ActivationLayer<T, LnCosh<T> >;

}  // namespace yannq

#endif
