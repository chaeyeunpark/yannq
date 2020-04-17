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
public:
	using VectorType = typename AbstractLayer<T>::VectorType;
	using MatrixType = typename AbstractLayer<T>::MatrixType;
	using VectorRefType = typename AbstractLayer<T>::VectorRefType;
	using VectorConstRefType = typename AbstractLayer<T>::VectorConstRefType;

private:
	bool useBias_;

	uint32_t inputDim_;        // input size
	uint32_t outputDim_;       // output size
	uint32_t npar_;           // number of parameters in layer

	MatrixType weight_;  // Weight parameters, W(in_size x out_size)
	VectorType bias_;    // Bias parameters, b(out_size x 1)
	// Note that input of this layer is also the output of
	// previous layer
	
public:
	/// Constructor
	FullyConnected(const uint32_t inputDim, const uint32_t outputDim,
			const bool useBias = false)
		: useBias_(useBias), inputDim_(inputDim), outputDim_(outputDim) 
	{
		weight_.resize(inputDim_, outputDim_);

		npar_ = inputDim_ * outputDim_;

		if (useBias_) {
			npar_ += outputDim_;
			bias_.resize(outputDim_);
		}
		bias_.setZero();
	}

	FullyConnected(const FullyConnected& ) = default;
	FullyConnected(FullyConnected&& ) = default;

	FullyConnected& operator=(const FullyConnected&) = default;
	FullyConnected& operator=(FullyConnected&&) = default;

	bool operator==(const FullyConnected& rhs) const
	{
		return (weight_ == rhs.weight_) && (bias_ == rhs.bias_);
	}


	std::string name() const override { return "Fully Connected Layer"; }

	nlohmann::json desc() const override 
	{
		nlohmann::json res;
		res["name"] = name();
		res["use_bias"] = useBias_;
		res["input_dim"] = inputDim_;
		res["output_dim"] = outputDim_;
		return res;
	}

	template<class RandomEngine>
	void randomizeParams(RandomEngine&& re, double sigma)
	{
		setParams(randomVector<T>(std::forward<RandomEngine>(re), sigma, npar_));
	}

	uint32_t paramDim() const override { return npar_; }

	uint32_t inputDim() const { return inputDim_; }

	uint32_t outputDim(uint32_t inputDim) const override {
		(void)inputDim;
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

	void updateParams(VectorConstRefType ups) override
	{
		if (useBias_)
		{
			bias_ += ups.head(outputDim_);
			weight_ += Eigen::Map<const MatrixType>(ups.data()+outputDim_, inputDim_, outputDim_);
		}
		else
		{
			weight_ += Eigen::Map<const MatrixType>(ups.data(), inputDim_, outputDim_);
		}
	}

	// Feedforward
	void forward(const VectorConstRefType& input, VectorRefType output) override 
	{
		if(useBias_)
			output = bias_;
		else
			output.setZero();
		output.noalias() += weight_.transpose() * input;
	}

	// Computes derivative.
	void backprop(const VectorConstRefType& prev_layer_output,
			const VectorConstRefType& /*this_layer_output*/,
			const VectorConstRefType& dout,
			VectorRefType din, VectorRefType der) override 
	{
		// dout = d(L) / d(z)
		// Derivative for bias, d(L) / d(b) = d(L) / d(z)
		int k = 0;

		if (useBias_) {
			der.segment(0, outputDim_) = dout;
			k += outputDim_;
		}

		// Derivative for weights, d(L) / d(W) = [d(L) / d(z)] * in'
		Eigen::Map<MatrixType> der_w{der.data() + k, inputDim_, outputDim_};

		der_w.noalias() = prev_layer_output * dout.transpose();

		// Compute d(L) / d_in = W * [d(L) / d(z)]
		din.noalias() = weight_ * dout;
	}

	uint32_t fanIn() override
	{
		return inputDim_;
	}
	uint32_t fanOut() override
	{
		return outputDim_;
	}
};

}  // namespace yannq

#endif
