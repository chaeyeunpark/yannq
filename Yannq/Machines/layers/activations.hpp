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
namespace activation
{
template<typename T>
class Identity 
{
	using ScalarType = T;
	using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
	using VectorRefType = Eigen::Ref<VectorType>;
	using VectorConstRefType = Eigen::Ref<const VectorType>;

public:
	static constexpr char name[] = "Identity";
	// A = Z
	inline void operator()(VectorConstRefType Z, VectorRefType A) const  {
		A.noalias() = Z;
	}

	// Apply the (derivative of activation function) matrix J to a vector F
	// A = Z
	// J = dA / dZ = I
	// G = J * F = F
	inline void ApplyJacobian(VectorConstRefType /*Z*/, VectorConstRefType /*A*/,
			VectorConstRefType F,
			VectorRefType G) const  {
		G.noalias() = F;
	}
};

template<typename T>
constexpr char Identity<T>::name[];

template<typename T>
class LnCosh 
{
	using ScalarType = T;
	using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
	using VectorRefType = Eigen::Ref<VectorType>;
	using VectorConstRefType = Eigen::Ref<const VectorType>;

public:
	constexpr static char name[] = "LnCosh";
	// A = Lncosh(Z)
	inline void operator()(VectorConstRefType Z, VectorRefType A) const  {
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
			VectorRefType G) const  {
		G.array() = F.array() * Z.array().tanh();
	}
};
template<typename T>
constexpr char LnCosh<T>::name[];

template<typename T>
class Tanh 
{
	using ScalarType = T;
	using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
	using VectorRefType = Eigen::Ref<VectorType>;
	using VectorConstRefType = Eigen::Ref<const VectorType>;

public:
	static constexpr char name[] = "Tanh";
	// A = Tanh(Z)
	inline void operator()(VectorConstRefType Z, VectorRefType A) const  {
		A.array() = Z.array().tanh();
	}

	// Apply the (derivative of activation function) matrix J to a vector F
	// A = Tanh(Z)
	// J = dA / dZ
	// G = J * F
	inline void ApplyJacobian(VectorConstRefType /*Z*/, VectorConstRefType A,
			VectorConstRefType F,
			VectorRefType G) const  {
		G.array() = F.array() * (1. - A.array() * A.array());
	}
};
template<typename T>
constexpr char Tanh<T>::name[];


template<typename T>
class Sigmoid
{
	static_assert(!is_complex_type<T>::value, "Sigmoid now only supports real parameters");

	using ScalarType = T;
	using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
	using VectorRefType = Eigen::Ref<VectorType>;
	using VectorConstRefType = Eigen::Ref<const VectorType>;

public:
	static constexpr char name[] = "Sigmoid";
	// A = Sigmoid(Z)
	inline void operator()(VectorConstRefType Z, VectorRefType A) const  {
		A.array() = Z.array().exp();
		A.array().inverse();
		A.array() += 1.;
		A.array().inverse();
	}

	// Apply the (derivative of activation function) matrix J to a vector F
	// A = Sigmoid(Z)
	// J = dA / dZ
	// G = J * F
	inline void ApplyJacobian(VectorConstRefType /*Z*/, VectorConstRefType A,
			VectorConstRefType F,
			VectorRefType G) const  {
		G.array() = (1. - A.array() ) * A.array();
	}
};
template<typename T>
constexpr char Sigmoid<T>::name[];


template<typename T, typename Enable = void>
class ReLU;

//for complex T
template<typename T>
class ReLU<T, std::enable_if_t<is_complex_type<T>::value > > 
{
	using ScalarType = T;
	using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
	using VectorRefType = Eigen::Ref<VectorType>;
	using VectorConstRefType = Eigen::Ref<const VectorType>;

	const double theta1_ = std::atan(1) * 3;
	const double theta2_ = -std::atan(1);

public:
	static constexpr char name[] = "ReLU";
	// A = Z
	inline void operator()(VectorConstRefType Z, VectorRefType A) const  {
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
			VectorRefType G) const  {
		for (int i = 0; i < Z.size(); ++i) {
			G(i) =
				(std::arg(Z(i)) < theta1_) && (std::arg(Z(i)) > theta2_) ? F(i) : 0.0;
		}
	}
};
//For real T
template<typename T>
class ReLU<T, std::enable_if_t<!is_complex_type<T>::value> >
{
	using ScalarType = T;
	using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
	using VectorRefType = Eigen::Ref<VectorType>;
	using VectorConstRefType = Eigen::Ref<const VectorType>;

public:
	static constexpr char name[] = "ReLU";
	// A = Z
	inline void operator()(VectorConstRefType Z, VectorRefType A) const  {
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
			VectorRefType G) const  
	{
		for (int i = 0; i < Z.size(); ++i) {
			G(i) = Z(i)>0?F(i):T{0.0};
		}
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
class LeakyReLU<T, std::enable_if_t<is_complex_type<T>::value>>
{
	using ScalarType = T;
	using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
	using VectorRefType = Eigen::Ref<VectorType>;
	using VectorConstRefType = Eigen::Ref<const VectorType>;

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
	inline void operator()(VectorConstRefType Z, VectorRefType A) const  {
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
			VectorRefType G) const  
	{
		for (int i = 0; i < Z.size(); ++i) {
			G(i) =
				(std::arg(Z(i)) < theta1_) && (std::arg(Z(i)) > theta2_) ? F(i) : negativeSlope_*F(i);
		}
	}

};
//For real T
template<typename T>
class LeakyReLU<T, std::enable_if_t<!is_complex_type<T>::value>>
{
	using ScalarType = T;
	using VectorType = Eigen::Matrix<T, Eigen::Dynamic, 1>;
	using VectorRefType = Eigen::Ref<VectorType>;
	using VectorConstRefType = Eigen::Ref<const VectorType>;
	
	const double negativeSlope_;

public:
	static constexpr char name[] = "LeakyReLU";

	LeakyReLU(const double negativeSlope = 0.3)
		: negativeSlope_{negativeSlope}
	{
	}
	// A = Z
	inline void operator()(VectorConstRefType Z, VectorRefType A) const  {
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
			VectorRefType G) const  
	{
		for (int i = 0; i < Z.size(); ++i) {
			G(i) = Z(i)>0?F(i):T{negativeSlope_*F(i)};
		}
	}
};
template<typename T>
constexpr char LeakyReLU<T, std::enable_if_t<!is_complex_type<T>::value>>::name[];
template<typename T>
constexpr char LeakyReLU<T, std::enable_if_t<is_complex_type<T>::value>>::name[];

}//namsepace activation

}// namespace yannq

#endif
