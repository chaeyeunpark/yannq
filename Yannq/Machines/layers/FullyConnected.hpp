// This file has been modified from the source file that is from NetKet proejct
// that is under Apache 2.0 Lisence. The original file header follows.
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

#ifndef YANNQ_FULLCONNLAYER_HH
#define YANNQ_FULLCONNLAYER_HH

#include <Eigen/Dense>
#include <complex>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include <Utilities/Utility.hpp>
#include "AbstractLayer.hpp"

namespace yannq {

template<typename T>
class FullyConnected : public AbstractLayer<T> 
{
	using VectorType = typename AbstractLayer<T>::VectorType;
	using MatrixType = typename AbstractLayer<T>::MatrixType;
	using VectorRefType = typename AbstractLayer<T>::VectorRefType;
	using VectorConstRefType = typename AbstractLayer<T>::VectorConstRefType;

	bool useBias_;

	uint32_t inputDim_;        // input size
	uint32_t outputDim_;       // output size
	uint32_t npar_;           // number of parameters in layer

	MatrixType weight_;  // Weight parameters, W(in_size x out_size)
	VectorType bias_;    // Bias parameters, b(out_size x 1)
	// Note that input of this layer is also the output of
	// previous layer

	constexpr static char name_[] = "Fully Connected Layer";

public:
	/// Constructor
	FullyConnected(const uint32_t input_size, const uint32_t output_size,
			const bool useBias = false)
		: useBias_(useBias), inputDim_(input_size), outputDim_(output_size) 
	{
		weight_.resize(inputDim_, outputDim_);
		bias_.resize(outputDim_);

		npar_ = inputDim_ * outputDim_;

		if (useBias_) {
			npar_ += outputDim_;
		} else {
			bias_.setZero();
		}

	}

	std::string name() const override { return name_; }

	nlohmann::json to_json() const override {
		nlohmann::json layerpar;
		layerpar["Name"] = "FullyConnected";
		layerpar["UseBias"] = useBias_;
		layerpar["Inputs"] = inputDim_;
		layerpar["Outputs"] = outputDim_;

		return layerpar;
	}

	template<class RandomEngine>
	void randomizeParams(RandomEngine&& re, double sigma)
	{
		setParams(randomVector<T>(std::forward<RandomEngine>(re), sigma, npar_));
	}

	uint32_t paramDim() const override { return npar_; }

	uint32_t inputDim() const { return inputDim_; }

	uint32_t outputDim(uint32_t inputDim) const override {
		assert(inputDim == inputDim_);
		return outputDim_; 
	}

	VectorType getParams() const override 
	{
		VectorType res(npar_);
		if (useBias_) {
			res.head(outputDim_) = Eigen::Map<const VectorType>(bias_.data(), outputDim_);
			res.segment(outputDim_, inputDim_*outputDim_) = Eigen::Map<const VectorType>(weight_.data(), inputDim_*outputDim_);
		}
		else
		{
			res = Eigen::Map<const VectorType>(weight_.data(), inputDim_*outputDim_); 
		}
		return res;
	}

	void setParams(VectorConstRefType pars) override 
	{
		if (useBias_)
		{
			bias_ = Eigen::Map<const VectorType>(pars.data(), outputDim_);
			weight_ = Eigen::Map<const MatrixType>(pars.data()+outputDim_, inputDim_, outputDim_);
		}
		else
		{
			weight_ = Eigen::Map<const MatrixType>(pars.data(), inputDim_, outputDim_);
		}
	}

	// Feedforward
	void forward(const VectorType &input, VectorType &output) override 
	{
		output = bias_;
		output.noalias() += weight_.transpose() * input;
	}

	// Computes derivative.
	void backprop(const VectorType &prev_layer_output,
			const VectorType & /*this_layer_output*/,
			const VectorType &dout, VectorType &din,
			VectorRefType der) override 
	{
		// dout = d(L) / d(z)
		// Derivative for bias, d(L) / d(b) = d(L) / d(z)
		int k = 0;

		if (useBias_) {
			Eigen::Map<VectorType> der_b{der.data() + k, outputDim_};

			der_b.noalias() = dout;
			k += outputDim_;
		}

		// Derivative for weights, d(L) / d(W) = [d(L) / d(z)] * in'
		Eigen::Map<MatrixType> der_w{der.data() + k, inputDim_, outputDim_};

		der_w.noalias() = prev_layer_output * dout.transpose();

		// Compute d(L) / d_in = W * [d(L) / d(z)]
		din.noalias() = weight_ * dout;
	}
};
template<typename T>
constexpr char FullyConnected<T>::name_[];
}  // namespace yannq

#endif
