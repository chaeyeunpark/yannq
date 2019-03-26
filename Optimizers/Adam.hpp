#ifndef NNQS_OPTIMIZERS_ADAM_HPP
#define NNQS_OPTIMIZERS_ADAM_HPP
#include <Eigen/Dense>
#include "Utilities/type_traits.hpp"

namespace nnqs
{

template<typename T>
class Adam
{
public:
	using RealVector = Eigen::Matrix<typename nnqs::remove_complex<T>::type, Eigen::Dynamic, 1>;
	using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

private:
	double alpha_;
	double beta1_;
	double beta2_;

	int t;
	RealVector m_;
	RealVector v_;
public:

	Adam(double alpha = 1e-3, double beta1 = 0.9, double beta2 = 0.999)
		: alpha_(alpha), beta1_(beta1), beta2_(beta2), t(0)
	{
	}

	
	typename std::enable_if<!nnqs::is_complex_type<T>::value, Vector>::type
	getUpdate(const Vector& grad)
	{
		const double eps = 1e-8;

		if(t ==0)
		{
			m_ = Vector::Zero(grad.rows());
			v_ = Vector::Zero(grad.rows());
		}
		++t;

		m_ *= beta1_;
		m_ += (1-beta1_)*grad;

		Vector g2= grad.unaryExpr([](T x){ return x*x;});
		v_ *= beta2_;
		v_ += (1-beta2_)*g2;

		Vector mHat = m_/(1.0-pow(beta1_, t));
		Vector vHat = v_/(1.0-pow(beta2_, t));

		Vector denom = vHat.unaryExpr([eps](T x){ return sqrt(x)+eps; });
		return -alpha_*mHat.cwiseQuotient(denom);
	}
	
	typename std::enable_if<nnqs::is_complex_type<T>::value, Vector>::type
	getUpdate(const Vector& grad)
	{
		const double eps = 1e-8;

		if(t ==0)
		{
			m_ = RealVector::Zero(2*grad.rows());
			v_ = RealVector::Zero(2*grad.rows());
		}
		++t;

		m_ *= beta1_;
		m_ += (1-beta1_)*grad;
	
		RealVector gradR = Eigen::Map<RealVector>(grad.data(), 2*grad.rows());

		RealVector g2= gradR.unaryExpr([](T x){ return x*x;});
		v_ *= beta2_;
		v_ += (1-beta2_)*g2;

		RealVector mHat = m_/(1.0-pow(beta1_, t));
		RealVector vHat = v_/(1.0-pow(beta2_, t));

		RealVector denom = vHat.unaryExpr([eps](T x){ return sqrt(x)+eps; });

		RealVector res = mHat.cwiseQuotient(denom);
		return -alpha_*Eigen::Map<Vector>(res.data(), grad.rows());
	}

};
} //namespace nnqs
#endif//NNQS_OPTIMIZERS_ADAM_HPP
