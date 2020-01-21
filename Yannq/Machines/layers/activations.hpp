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

#ifndef YANNQ_ACTIVATIONS_HPP
#define YANNQ_ACTIVATIONS_HPP

#include <Eigen/Dense>
#include <complex>
#include <iostream>
#include <random>
#include <vector>

#include <Utilities/type_traits.hpp>
#include <Utilities/Utility.hpp>

#include <nlohmann/json.hpp>

namespace yannq {

/**
  Abstract class for Activations.
*/
template <typename T>
class AbstractActivation {
public:
	using ScalarType = T;
	using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
	using VectorRefType = Eigen::Ref<VectorType>;
	using VectorConstRefType = Eigen::Ref<const VectorType>;

	virtual void operator()(VectorConstRefType Z, VectorRefType A) const = 0;

	// Z is the layer output before applying nonlinear function
	// A = nonlinearfunction(Z)
	// F = dL/dA is the derivative of A wrt the output L = log(psi(v))
	// G is the place to write the output i.e. G = dL/dZ = dL/dA * dA/dZ
	virtual void ApplyJacobian(VectorConstRefType Z, VectorConstRefType A,
			VectorConstRefType F, VectorRefType G) const = 0;
	virtual ~AbstractActivation() {}

	virtual nlohmann::json to_json() const = 0;
};


template<typename T>
class Identity : public AbstractActivation<T> {
	using VectorType = typename AbstractActivation<T>::VectorType;
	using VectorRefType = typename AbstractActivation<T>::VectorRefType;
	using VectorConstRefType = typename AbstractActivation<T>::VectorConstRefType;


public:
	static constexpr char name[] = "Identity";
	// A = Z
	inline void operator()(VectorConstRefType Z, VectorRefType A) const override {
		A.noalias() = Z;
	}

	// Apply the (derivative of activation function) matrix J to a vector F
	// A = Z
	// J = dA / dZ = I
	// G = J * F = F
	inline void ApplyJacobian(VectorConstRefType /*Z*/, VectorConstRefType /*A*/,
			VectorConstRefType F,
			VectorRefType G) const override {
		G.noalias() = F;
	}

	nlohmann::json to_json() const override
	{
		return nlohmann::json
		{
			{"name", name}
		};
	}
};
template<typename T>
constexpr char Identity<T>::name[];

template<typename T>
class LnCosh : public AbstractActivation<T> {
	using VectorType = typename AbstractActivation<T>::VectorType;
	using VectorRefType = typename AbstractActivation<T>::VectorRefType;
	using VectorConstRefType = typename AbstractActivation<T>::VectorConstRefType;

public:
	constexpr static char name[] = "LnCosh";
	// A = Lncosh(Z)
	inline void operator()(VectorConstRefType Z, VectorRefType A) const override {
		for (int i = 0; i < A.size(); ++i) {
			A(i) = logCosh(Z(i));
		}
	}

	// Apply the (derivative of activation function) matrix J to a vector F
	// A = Lncosh(Z)
	// J = dA / dZ
	// G = J * F
	inline void ApplyJacobian(VectorConstRefType Z, VectorConstRefType /*A*/,
			VectorConstRefType F,
			VectorRefType G) const override {
		G.array() = F.array() * Z.array().tanh();
	}
	nlohmann::json to_json() const override
	{
		return nlohmann::json
		{
			{"name", name}
		};
	}
};
template<typename T>
constexpr char LnCosh<T>::name[];

template<typename T>
class Tanh : public AbstractActivation<T> {
	using VectorType = typename AbstractActivation<T>::VectorType;
	using VectorRefType = typename AbstractActivation<T>::VectorRefType;
	using VectorConstRefType = typename AbstractActivation<T>::VectorConstRefType;

public:
	static constexpr char name[] = "Tanh";
	// A = Tanh(Z)
	inline void operator()(VectorConstRefType Z, VectorRefType A) const override {
		A.array() = Z.array().tanh();
	}

	// Apply the (derivative of activation function) matrix J to a vector F
	// A = Tanh(Z)
	// J = dA / dZ
	// G = J * F
	inline void ApplyJacobian(VectorConstRefType /*Z*/, VectorConstRefType A,
			VectorConstRefType F,
			VectorRefType G) const override {
		G.array() = F.array() * (1 - A.array() * A.array());
	}
	nlohmann::json to_json() const override
	{
		return nlohmann::json
		{
			{"name", name}
		};
	}
};
template<typename T>
constexpr char Tanh<T>::name[];

template<typename T, typename Enable = void>
class ReLU;

//for complex T
template<typename T>
class ReLU<T, std::enable_if_t<is_complex_type<T>::value > > : public AbstractActivation<T> {
	using VectorType = typename AbstractActivation<T>::VectorType;
	using VectorRefType = typename AbstractActivation<T>::VectorRefType;
	using VectorConstRefType = typename AbstractActivation<T>::VectorConstRefType;

	const double theta1_ = std::atan(1) * 3;
	const double theta2_ = -std::atan(1);

public:
	static constexpr char name[] = "ReLU";
	// A = Z
	inline void operator()(VectorConstRefType Z, VectorRefType A) const override {
		for (int i = 0; i < Z.size(); ++i) {
			A(i) =
				(std::arg(Z(i)) < theta1_) && (std::arg(Z(i)) > theta2_) ? Z(i) : 0.0;
		}
	}

	// Apply the (derivative of activation function) matrix J to a vector F
	// A = Z
	// J = dA / dZ = I
	// G = J * F = F
	inline void ApplyJacobian(VectorConstRefType Z, VectorConstRefType /*A*/,
			VectorConstRefType F,
			VectorRefType G) const override {
		for (int i = 0; i < Z.size(); ++i) {
			G(i) =
				(std::arg(Z(i)) < theta1_) && (std::arg(Z(i)) > theta2_) ? F(i) : 0.0;
		}
	}
	nlohmann::json to_json() const override
	{
		return nlohmann::json
		{
			{"name", name}
		};
	}
};
//For real T
template<typename T>
class ReLU<T, std::enable_if_t<!is_complex_type<T>::value> >
	: public AbstractActivation<T> 
{
	using VectorType = typename AbstractActivation<T>::VectorType;
	using VectorRefType = typename AbstractActivation<T>::VectorRefType;
	using VectorConstRefType = typename AbstractActivation<T>::VectorConstRefType;

public:
	static constexpr char name[] = "ReLU";
	// A = Z
	inline void operator()(VectorConstRefType Z, VectorRefType A) const override {
		for (int i = 0; i < Z.size(); ++i) {
			A(i) = std::max(Z(i), T{0.0});
		}
	}

	// Apply the (derivative of activation function) matrix J to a vector F
	// A = Z
	// J = dA / dZ = I
	// G = J * F = F
	inline void ApplyJacobian(VectorConstRefType Z, VectorConstRefType /*A*/,
			VectorConstRefType F,
			VectorRefType G) const override 
	{
		for (int i = 0; i < Z.size(); ++i) {
			G(i) = Z(i)>0?F(i):T{0.0};
		}
	}
	nlohmann::json to_json() const override
	{
		return nlohmann::json
		{
			{"name", name}
		};
	}

};

template<typename T>
constexpr char ReLU<T, std::enable_if_t<!is_complex_type<T>::value>>::name[];
template<typename T>
constexpr char ReLU<T, std::enable_if_t<is_complex_type<T>::value>>::name[];


template<typename T, typename Enable = void>
class LeakyReLU;

//for complex T
template<typename T>
class LeakyReLU<T, std::enable_if_t<is_complex_type<T>::value>> : public AbstractActivation<T> {
	using VectorType = typename AbstractActivation<T>::VectorType;
	using VectorRefType = typename AbstractActivation<T>::VectorRefType;
	using VectorConstRefType = typename AbstractActivation<T>::VectorConstRefType;

	const double theta1_ = std::atan(1) * 3;
	const double theta2_ = -std::atan(1);
	const double negativeSlope_ = 0.3;

public:
	static constexpr char name[] = "LeakyReLU";

	LeakyReLU(const double negativeSlope = 0.3)
		: negativeSlope_{negativeSlope}
	{
	}

	// A = Z
	inline void operator()(VectorConstRefType Z, VectorRefType A) const override {
		for (int i = 0; i < Z.size(); ++i) {
			A(i) =
				(std::arg(Z(i)) < theta1_) && (std::arg(Z(i)) > theta2_) ? Z(i) : negativeSlope_*Z(i);
		}
	}

	// Apply the (derivative of activation function) matrix J to a vector F
	// A = Z
	// J = dA / dZ = I
	// G = J * F = F
	inline void ApplyJacobian(VectorConstRefType Z, VectorConstRefType /*A*/,
			VectorConstRefType F,
			VectorRefType G) const override 
	{
		for (int i = 0; i < Z.size(); ++i) {
			G(i) =
				(std::arg(Z(i)) < theta1_) && (std::arg(Z(i)) > theta2_) ? F(i) : negativeSlope_*F(i);
		}
	}

	nlohmann::json to_json() const override
	{
		return nlohmann::json
		{
			{"name", name}
		};
	}

};
//For real T
template<typename T>
class LeakyReLU<T, std::enable_if_t<!is_complex_type<T>::value>> : public AbstractActivation<T> 
{
	using VectorType = typename AbstractActivation<T>::VectorType;
	using VectorRefType = typename AbstractActivation<T>::VectorRefType;
	using VectorConstRefType = typename AbstractActivation<T>::VectorConstRefType;
	
	const double negativeSlope_;

public:
	static constexpr char name[] = "LeakyReLU";

	LeakyReLU(const double negativeSlope = 0.3)
		: negativeSlope_{negativeSlope}
	{
	}
	// A = Z
	inline void operator()(VectorConstRefType Z, VectorRefType A) const override {
		for (int i = 0; i < Z.size(); ++i) {
			A(i) = std::max(Z(i), T{negativeSlope_*Z(i)});
		}
	}

	// Apply the (derivative of activation function) matrix J to a vector F
	// A = Z
	// J = dA / dZ = I
	// G = J * F = F
	inline void ApplyJacobian(VectorConstRefType Z, VectorConstRefType /*A*/,
			VectorConstRefType F,
			VectorRefType G) const override 
	{
		for (int i = 0; i < Z.size(); ++i) {
			G(i) = Z(i)>0?F(i):T{negativeSlope_*F(i)};
		}
	}

	nlohmann::json to_json() const override
	{
		return nlohmann::json
		{
			{"name", name}
		};
	}
};
template<typename T>
constexpr char LeakyReLU<T, std::enable_if_t<!is_complex_type<T>::value>>::name[];
template<typename T>
constexpr char LeakyReLU<T, std::enable_if_t<is_complex_type<T>::value>>::name[];


}// namespace yannq

#endif
