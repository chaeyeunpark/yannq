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

#pragma once

#include <complex>
#include <fstream>
#include <random>
#include <vector>

#include <Eigen/Dense>
#include "nlohmann/json.hpp"

namespace yannq {
/**
  Abstract class for Neural Network layer.
*/

template<typename T>
class AbstractLayer {
public:
	using Scalar = T;
	using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
	using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
	using VectorRef = Eigen::Ref<Vector>;
	using VectorConstRef = Eigen::Ref<const Vector>;

	/**
	  Member function returning the name of the layer.
	  @return Name of Layer.
	  */
	virtual std::string name() const = 0;

	/**
	  Member function returning the number of variational parameters.
	  @return Number of variational parameters in the Layer.
	  */
	virtual uint32_t paramDim() const = 0;

	virtual uint32_t fanIn() { return 0; }
	virtual uint32_t fanOut() { return 0; }
	
	virtual uint32_t outputDim(uint32_t inputDim) const = 0;

	/**
	  Member function writing the current set of parameters in the machine.
	  @param pars is where the layer parameters are written into.
	  @param start_idx is the index of the vector pars to start writing from.
	  */
	virtual Vector getParams() const = 0;

	/**
	  Member function setting the current set of parameters in the layer.
	  @param pars is where the layer parameters are to be read from.
	  @param start_idx is the index of the vector pars to start reading from.
	  */
	virtual void setParams(VectorConstRef pars) = 0;

	virtual void updateParams(VectorConstRef pars) { (void)pars; return ; }


	/**
	  Member function to feedforward through the layer.
	  @param input a constant reference to the input to the layer
	  @param output reference to the output vector. Must have the size of outputDim. 
	  */
	virtual void forward(const VectorConstRef& input, VectorRef output) = 0;

	/**
	  Member function to perform backpropagation to compute derivates.
	  Internal stride of din and der must be 1.

	  @param prev_layer_output a constant reference to the output from previous
	  layer.
	  @param this_layer_output a constant reference to the output from the current
	  layer.
	  @param next_layer_data a constant reference to the derivative dL/dA where A is
	  the activations of the current layer and L is the the final output of the 
	  Machine: L = log(psi(v))
	  @param din a constant reference to the derivative of the input from the
	  current layer.
	  @param der a constant reference to the derivatives wrt to the parameters in
	  the machine.
	*/
	virtual void backprop(const VectorConstRef& prev_layer_output,
			const VectorConstRef& this_layer_output,
			const VectorConstRef& dout, 
			VectorRef din,
			VectorRef der) = 0;

	virtual nlohmann::json desc() const = 0;

	/**
	  destructor
	  */
	virtual ~AbstractLayer() {}
};
}  // namespace yannq
