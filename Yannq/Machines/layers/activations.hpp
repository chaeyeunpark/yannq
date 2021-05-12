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
public:
	using Scalar = T;
	using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
	using VectorRef = Eigen::Ref<Vector>;
	using VectorConstRef = Eigen::Ref<const Vector>;

	// A = Z
	inline void operator()(VectorConstRef Z, VectorRef A) const  {
		A.noalias() = Z;
	}

	// Apply the (derivative of activation function) matrix J to a vector F
	// A = Z
	// J = dA / dZ = I
	// G = J * F = F
	inline void ApplyJacobian(VectorConstRef /*Z*/, VectorConstRef /*A*/,
			VectorConstRef F,
			VectorRef G) const  {
		G.noalias() = F;
	}

	std::string name() const {
		return "Identity";
	}

	template<class Archive>
	void serialize(Archive& /*ar*/)
	{
	}
};


template<typename T>
class LnCosh 
{
public:
	using Scalar = T;
	using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
	using VectorRef = Eigen::Ref<Vector>;
	using VectorConstRef = Eigen::Ref<const Vector>;

	// A = Lncosh(Z)
	inline void operator()(VectorConstRef Z, VectorRef A) const  {
		for (int i = 0; i < A.size(); ++i) {
			A(i) = logCosh(Z(i));
		}
	}

	// Apply the (derivative of activation function) matrix J to a vector F
	// A = Lncosh(Z)
	// J = dA / dZ
	// G = J * F
	inline void ApplyJacobian(VectorConstRef Z, VectorConstRef /*A*/,
			VectorConstRef F,
			VectorRef G) const  {
		G.array() = F.array() * Z.array().tanh();
	}

	std::string name() const {
		return "LnCosh";
	}

	template<class Archive>
	void serialize(Archive& /*ar*/)
	{
	}
};

template<typename T>
class Tanh 
{
public:	
	using Scalar = T;
	using RealScalar = remove_complex_t<Scalar>;
	using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
	using VectorRef = Eigen::Ref<Vector>;
	using VectorConstRef = Eigen::Ref<const Vector>;

private:
	RealScalar constant_;

public:

	explicit Tanh(const RealScalar constant = 1.0) //constant*tanh(z/constant)
		: constant_{constant}
	{
	}

	// A = a*Tanh(Z/a)
	inline void operator()(VectorConstRef Z, VectorRef A) const  {
		A.array() = constant_ * (Z.array() / constant_).tanh();
	}

	// Apply the (derivative of activation function) matrix J to a vector F
	// A = a*Tanh(Z/a)
	// J = dA / dZ
	// G = J * F
	inline void ApplyJacobian(VectorConstRef /*Z*/, VectorConstRef A,
			VectorConstRef F,
			VectorRef G) const  {
		G.array() = F.array() * (1. - A.array() * A.array() / (constant_*constant_));
	}

	std::string name() const
	{
		return "Tanh";
	}

	template<class Archive>
	void serialize(Archive& ar)
	{
		ar(constant_);
	}
};

template<typename T>
class WeakTanh
{
public:	
	using Scalar = T;
	using RealScalar = remove_complex_t<Scalar>;
	using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
	using VectorRef = Eigen::Ref<Vector>;
	using VectorConstRef = Eigen::Ref<const Vector>;

private:

public:

	explicit WeakTanh() //x - tanh(x)/2
	{
	}

	// A = a*Tanh(Z/a)
	inline void operator()(VectorConstRef Z, VectorRef A) const  {
		A.array() = Z.array();
		A.array() -= Z.array().tanh()/2.0;
	}

	// Apply the (derivative of activation function) matrix J to a vector F
	// A = Tanh(Z)
	// J = dA / dZ
	// G = J * F
	inline void ApplyJacobian(VectorConstRef Z, VectorConstRef /*A*/,
			VectorConstRef F,
			VectorRef G) const  {
		Eigen::Array<T, Eigen::Dynamic, 1> tanh = Z.array().tanh();
		G.array() = F.array() * (1.0 +  tanh * tanh)/2.0;
	}

	std::string name() const
	{
		return "WeakTanh";
	}

	template<class Archive>
	void serialize(Archive& /*ar*/)
	{
	}
};


template<typename T>
class Sigmoid
{
	static_assert(!is_complex_type<T>::value, "Sigmoid now only supports real parameters");
public:
	using Scalar = T;
	using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
	using VectorRef = Eigen::Ref<Vector>;
	using VectorConstRef = Eigen::Ref<const Vector>;

	// A = Sigmoid(Z)
	inline void operator()(VectorConstRef Z, VectorRef A) const  {
		A.array() = Z.array().exp();
		A.array().inverse();
		A.array() += 1.;
		A.array().inverse();
	}

	// Apply the (derivative of activation function) matrix J to a vector F
	// A = Sigmoid(Z)
	// J = dA / dZ
	// G = J * F
	inline void ApplyJacobian(VectorConstRef /*Z*/, VectorConstRef A,
			VectorConstRef /*F*/,
			VectorRef G) const  {
		G.array() = (1. - A.array() ) * A.array();
	}

	std::string name() const
	{
		return "Sigmoid";
	}

	template<class Archive>
	void serialize(Archive& /*ar*/)
	{
	}
};

template<typename T, typename Enable = void>
class ReLU;

//for complex T
template<typename T>
class ReLU<T, typename std::enable_if<is_complex_type<T>::value>::type > 
{
public:
	using Scalar = T;
	using RealScalar = remove_complex_t<Scalar>;
	using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
	using VectorRef = Eigen::Ref<Vector>;
	using VectorConstRef = Eigen::Ref<const Vector>;

	constexpr static RealScalar theta1_ = 3*M_PI/4;
	constexpr static RealScalar theta2_ = -M_PI/4;

	// A = ReLU(Z)
	inline void operator()(VectorConstRef Z, VectorRef A) const  {
		for (int i = 0; i < Z.size(); ++i) {
			A(i) =
				(std::arg(Z(i)) < theta1_) && (std::arg(Z(i)) > theta2_) ? Z(i) : 0.0;
		}
	}

	// Apply the (derivative of activation function) matrix J to a vector F
	// A = ReLU(Z)
	// J = dA / dZ 
	// G = J * F = F
	inline void ApplyJacobian(VectorConstRef Z, VectorConstRef /*A*/,
			VectorConstRef F,
			VectorRef G) const  {
		for (int i = 0; i < Z.size(); ++i) {
			G(i) =
				(std::arg(Z(i)) < theta1_) && (std::arg(Z(i)) > theta2_) ? F(i) : 0.0;
		}
	}

	std::string name() const
	{
		return "ReLU";
	}

	template<class Archive>
	void serialize(Archive& /*ar*/)
	{
	}
};
//For real T
template<typename T>
class ReLU<T, typename std::enable_if<!is_complex_type<T>::value>::type >
{
public:
	using Scalar = T;
	using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
	using VectorRef = Eigen::Ref<Vector>;
	using VectorConstRef = Eigen::Ref<const Vector>;

	// A = ReLU(Z)
	inline void operator()(VectorConstRef Z, VectorRef A) const  {
		for (int i = 0; i < Z.size(); ++i) {
			A(i) = std::max(Z(i), T{0.0});
		}
	}

	// Apply the (derivative of activation function) matrix J to a vector F
	// A = ReLU(Z)
	// J = dA / dZ 
	// G = J * F = F
	inline void ApplyJacobian(VectorConstRef Z, VectorConstRef /*A*/,
			VectorConstRef F,
			VectorRef G) const  
	{
		for (int i = 0; i < Z.size(); ++i) {
			G(i) = Z(i)>0?F(i):T{0.0};
		}
	}

	std::string name() const
	{
		return "ReLU";
	}

	template<class Archive>
	void serialize(Archive& /*ar*/)
	{
	}
};

template<typename T, typename Enable = void>
class LeakyReLU;

//for complex T
template<typename T>
class LeakyReLU<T, typename std::enable_if<is_complex_type<T>::value>::type>
{
public:
	using Scalar = T;
	using RealScalar = remove_complex_t<Scalar>;
	using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
	using VectorRef = Eigen::Ref<Vector>;
	using VectorConstRef = Eigen::Ref<const Vector>;

	constexpr static RealScalar theta1_ = 3*M_PI/4;
	constexpr static RealScalar theta2_ = -M_PI/4;
	RealScalar negativeSlope_;

	LeakyReLU(const RealScalar negativeSlope = 0.1)
		: negativeSlope_{negativeSlope}
	{
	}

	// A = LeakyReLU(Z)
	inline void operator()(VectorConstRef Z, VectorRef A) const  {
		for (int i = 0; i < Z.size(); ++i) {
			A(i) =
				(std::arg(Z(i)) < theta1_) && (std::arg(Z(i)) > theta2_) ? Z(i) : negativeSlope_*Z(i);
		}
	}

	// Apply the (derivative of activation function) matrix J to a vector F
	// A = LeakyReLU(Z)
	// J = dA / dZ 
	// G = J * F = F
	inline void ApplyJacobian(VectorConstRef Z, VectorConstRef /*A*/,
			VectorConstRef F,
			VectorRef G) const  
	{
		for (int i = 0; i < Z.size(); ++i) {
			G(i) =
				(std::arg(Z(i)) < theta1_) && (std::arg(Z(i)) > theta2_) ? F(i) : negativeSlope_*F(i);
		}
	}

	std::string name() const
	{
		return "LeakyReLU";
	}

	template<class Archive>
	void serialize(Archive& ar)
	{
		ar(negativeSlope_);
	}
};

//For real T
template<typename T>
class LeakyReLU<T, typename std::enable_if<!is_complex_type<T>::value>::type>
{
public:
	using Scalar = T;
	using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
	using VectorRef = Eigen::Ref<Vector>;
	using VectorConstRef = Eigen::Ref<const Vector>;
	
	Scalar negativeSlope_;

	LeakyReLU(const Scalar negativeSlope = 0.1)
		: negativeSlope_{negativeSlope}
	{
	}
	// A = LeakyReLU(Z)
	inline void operator()(VectorConstRef Z, VectorRef A) const  {
		for (int i = 0; i < Z.size(); ++i) {
			A(i) = Z(i)>0?Z(i):T{negativeSlope_*Z(i)};
		}
	}

	// Apply the (derivative of activation function) matrix J to a vector F
	// A = LeakyReLU(Z)
	// J = dA / dZ 
	// G = J * F = F
	inline void ApplyJacobian(VectorConstRef Z, VectorConstRef /*A*/,
			VectorConstRef F,
			VectorRef G) const  
	{
		for (int i = 0; i < Z.size(); ++i) {
			G(i) = Z(i)>0?F(i):T{negativeSlope_*F(i)};
		}
	}

	std::string name() const
	{
		return "LeakyReLU";
	}
	template<class Archive>
	void serialize(Archive& ar)
	{
		ar(negativeSlope_);
	}
};


//For real T
template<typename T>
class HardTanh
{
public:
	static_assert(!is_complex_type<T>::value, "T must be a real type for HardTanh");
	using Scalar = T;
	using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
	using VectorRef = Eigen::Ref<Vector>;
	using VectorConstRef = Eigen::Ref<const Vector>;
	
	HardTanh()
	{
	}

	// A = hardtanh(Z)
	inline void operator()(VectorConstRef Z, VectorRef A) const  
	{
		using std::abs;
		for (int i = 0; i < Z.size(); ++i) 
		{
			A(i) = abs(Z(i))>1.0?copysign(1.0, Z(i)):Z(i);
		}
	}

	// Apply the (derivative of activation function) matrix J to a vector F
	// A = hardtanh(Z)
	// J = dA / dZ 
	// G = J * F = F
	inline void ApplyJacobian(VectorConstRef Z, VectorConstRef /*A*/,
			VectorConstRef F, VectorRef G) const  
	{
		using std::abs;
		for (int i = 0; i < Z.size(); ++i) {
			G(i) = abs(Z(i))>1.0?0.0:F(i);
		}
	}

	std::string name() const
	{
		return "HardTanh";
	}

	template<class Archive>
	void serialize(Archive& /*ar*/)
	{
	}
};

template<typename T>
class SoftShrink
{
public:
	static_assert(!is_complex_type<T>::value, "T must be a real type for SoftShrink");
	using Scalar = T;
	using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
	using VectorRef = Eigen::Ref<Vector>;
	using VectorConstRef = Eigen::Ref<const Vector>;
	
	SoftShrink()
	{
	}

	// A = softshrink(Z)
	inline void operator()(VectorConstRef Z, VectorRef A) const  
	{
		using std::abs;
		for (int i = 0; i < Z.size(); ++i) 
		{
			A(i) = abs(Z(i))<1.0?0.0:(Z(i)-copysign(1.0,Z(i)));
		}
	}

	// Apply the (derivative of activation function) matrix J to a vector F
	// A = softshrink(Z)
	// J = dA / dZ 
	// G = J * F = F
	inline void ApplyJacobian(VectorConstRef Z, VectorConstRef /*A*/,
			VectorConstRef F, VectorRef G) const  
	{
		using std::abs;
		for (int i = 0; i < Z.size(); ++i) {
			G(i) = abs(Z(i))<1.0?0.0:F(i);
		}
	}

	std::string name() const
	{
		return "SoftShrink";
	}

	template<class Archive>
	void serialize(Archive& /*ar*/)
	{
	}

};

template<typename T>
class LeakyHardTanh
{
public:
	static_assert(!is_complex_type<T>::value, "T must be a real type for LeakyHardTanh");
	using Scalar = T;
	using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
	using VectorRef = Eigen::Ref<Vector>;
	using VectorConstRef = Eigen::Ref<const Vector>;

	Scalar outsideSlope_;
	
	LeakyHardTanh(Scalar outsideSlope = 0.01)
		: outsideSlope_{outsideSlope}
	{
	}

	// A = leakyhardtanh(Z)
	inline void operator()(VectorConstRef Z, VectorRef A) const  
	{
		using std::abs;
		for (int i = 0; i < Z.size(); ++i) 
		{
			T val = abs(Z(i))>1.0?copysign(1.0, Z(i)):Z(i);
			val += abs(Z(i))<1.0?0.0:outsideSlope_*(Z(i)-copysign(1.0,Z(i)));
			A(i) = val;
		}
	}

	// Apply the (derivative of activation function) matrix J to a vector F
	// A = leakyhardtanh(Z)
	// J = dA / dZ 
	// G = J * F = F
	inline void ApplyJacobian(VectorConstRef Z, VectorConstRef /*A*/,
			VectorConstRef F, VectorRef G) const  
	{
		using std::abs;
		for (int i = 0; i < Z.size(); ++i) {
			G(i) = abs(Z(i))>1.0?outsideSlope_*F(i):F(i);
		}
	}

	std::string name() const
	{
		return "LeakyHardTanh";
	}

	template<class Archive>
	void serialize(Archive& ar)
	{
		ar(outsideSlope_);
	}
};


template<typename T>
class SoftSign
{
public:
	static_assert(!is_complex_type<T>::value, "T must be a real type for SoftSign");
	using Scalar = T;
	using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
	using VectorRef = Eigen::Ref<Vector>;
	using VectorConstRef = Eigen::Ref<const Vector>;
	
	SoftSign()
	{
	}

	// A = softsign(Z)
	inline void operator()(VectorConstRef Z, VectorRef A) const  
	{
		A.array() = Z.array()/(Z.array().abs()+1.0);
	}

	// Apply the (derivative of activation function) matrix J to a vector F
	// A = softsign(Z)
	// J = dA / dZ = I
	// G = J * F = F
	inline void ApplyJacobian(VectorConstRef Z, VectorConstRef /*A*/,
			VectorConstRef F, VectorRef G) const  
	{
		G.array() = (1.0+Z.array().abs()).square().inverse()*F.array();
	}

	std::string name() const
	{
		return "SoftSign";
	}

	template<class Archive>
	void serialize(Archive& /*ar*/)
	{
	}
};

template<typename T>
class Cos
{
private:
	T a_;
public:
	static_assert(!is_complex_type<T>::value, "T must be a real type for Cos");
	using Scalar = T;
	using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
	using VectorRef = Eigen::Ref<Vector>;
	using VectorConstRef = Eigen::Ref<const Vector>;

	
	Cos(T a = 1.0)
		: a_{a}
	{
	}

	// A = cos(Z)
	inline void operator()(VectorConstRef Z, VectorRef A) const  
	{
		A.array() = a_*cos(Z.array()*M_PI/a_);
	}

	// Apply the (derivative of activation function) matrix J to a vector F
	// A = a*cos(\pi Z/a)
	// J = dA / dZ = \pi sin(\pi Z/ a)
	// G = J * F = F
	inline void ApplyJacobian(VectorConstRef Z, VectorConstRef /*A*/,
			VectorConstRef F, VectorRef G) const  
	{
		G.array() = -M_PI*sin(M_PI*Z.array()/a_)*F.array();
	}

	std::string name() const
	{
		return "Cos";
	}

	template<class Archive>
	void serialize(Archive& ar)
	{
		ar(a_);
	}
};
}//namsepace activation

}// namespace yannq

#endif
