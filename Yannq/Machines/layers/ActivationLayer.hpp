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

#include <complex>
#include <fstream>
#include <random>
#include <vector>
#include <tuple>
#include <type_traits>

#include <Eigen/Dense>

#include "AbstractLayer.hpp"
#include "activations.hpp"

/**
 * Misc: Because of bugs in Intel C compiler (ICC), constexpr static member 
 * variable of template class does not work. If there's any update in ICC 
 * regarding this bug, one may update name function to return such a member
 * variable.
 * */

namespace yannq {

template <typename T, class Activation>
class ActivationLayer
	: public AbstractLayer<T> 
{
public:
	using Scalar = T;
	using Vector = typename AbstractLayer<T>::Vector;
	using Matrix = typename AbstractLayer<T>::Matrix;
	using VectorRef = typename AbstractLayer<T>::VectorRef;
	using VectorConstRef = typename AbstractLayer<T>::VectorConstRef;

private:

	Activation f_;  

public:
	ActivationLayer()
	{
	}

	template<typename ...Ts, 
		typename disable_if<
            std::is_same<
                typename std::remove_reference<typename get_nth_type<0, Ts...>::type>::type,
                ActivationLayer<T, Activation>
			>::value, int>::type = 0>
	ActivationLayer(Ts&&... args)
		: f_(std::forward<Ts>(args)...)
	{
	}

	ActivationLayer(const ActivationLayer&) = default;
	ActivationLayer(ActivationLayer&&) = default;
	
	ActivationLayer& operator=(const ActivationLayer&) = default;
	ActivationLayer& operator=(ActivationLayer&&) = default;
	
	std::string name() const override 
	{
		return std::string("Activation Layer");
	}
	
	bool operator==(const ActivationLayer& rhs) const
	{
		return f_.name() == rhs.f_.name();
	}

	uint32_t paramDim() const override { return 0; }

	Vector getParams() const override
	{
		return Vector{};
	}

	void setParams(VectorConstRef pars) override
	{
		(void)pars;
	}

	uint32_t outputDim(uint32_t inputDim) const override 
	{
		return inputDim; 
	}

	// Feedforward
	void forward(const VectorConstRef& input, VectorRef output) override 
	{
		assert(input.size() == output.size());
		f_.operator()(input, output);
	}

	// Computes derivative.
	void backprop(const VectorConstRef& prev_layer_output,
			const VectorConstRef& this_layer_output,
			const VectorConstRef& dout,
			VectorRef din, VectorRef /*der*/) override 
	{
		din.resize(prev_layer_output.size());
		f_.ApplyJacobian(prev_layer_output, this_layer_output, dout, din);
	}

	nlohmann::json desc() const override
	{
		nlohmann::json layerpar;
		layerpar["name"] = name();
		layerpar["activation"] = f_.name();

		return layerpar;
	}

	template<class Archive>
	void serialize(Archive& ar)
	{
		ar(f_);
	}
};

template<typename T>
using Identity = ActivationLayer<T, activation::Identity<T> >;

template<typename T>
using LnCosh = ActivationLayer<T, activation::LnCosh<T> >;

template<typename T>
using Tanh = ActivationLayer<T, activation::Tanh<T> >;

template<typename T>
using WeakTanh = ActivationLayer<T, activation::WeakTanh<T> >;

template<typename T>
using Sigmoid = ActivationLayer<T, activation::Sigmoid<T> >;

template<typename T>
using ReLU = ActivationLayer<T, activation::ReLU<T> >;

template<typename T>
using LeakyReLU = ActivationLayer<T, activation::LeakyReLU<T> >;

template<typename T>
using HardTanh = ActivationLayer<T, activation::HardTanh<T> >;

template<typename T>
using SoftShrink = ActivationLayer<T, activation::SoftShrink<T> >;

template<typename T>
using LeakyHardTanh = ActivationLayer<T, activation::LeakyHardTanh<T> >;

template<typename T>
using SoftSign = ActivationLayer<T, activation::SoftSign<T> >;

template<typename T>
using Cos = ActivationLayer<T, activation::Cos<T> >;

}  // namespace yannq

#endif
