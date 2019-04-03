#ifndef NNQS_OPTIMIZERS_ADAM_HPP
#define NNQS_OPTIMIZERS_ADAM_HPP
#include <Eigen/Dense>
#include <type_traits>
#include "Utilities/type_traits.hpp"
#include "Optimizers/Optimizer.hpp"

namespace nnqs
{

template <typename T>
class AdamBase
	: public Optimizer<T>
{
public:
	using typename Optimizer<T>::Vector;
	using typename Optimizer<T>::RealVector;

protected:
	double alpha_;
	double beta1_;
	double beta2_;

public:
	static constexpr double DEFAULT_PARAMS[] = {1e-3, 0.9, 0.999};

	static nlohmann::json defaultParams()
	{
		return nlohmann::json
		{
			{"name", "Adam"},
			{"alhpa", DEFAULT_PARAMS[0]},
			{"beta1", DEFAULT_PARAMS[1]},
			{"beta2", DEFAULT_PARAMS[2]}
		};
	}

	nlohmann::json params() const override
	{
		return nlohmann::json
		{
			{"name", "Adam"},
			{"alhpa", alpha_},
			{"beta1", beta1_},
			{"beta2", beta2_}
		};
	}

	AdamBase(const nlohmann::json& params)
		: alpha_(params.value("alpha", DEFAULT_PARAMS[0])), 
			beta1_(params.value("beta1", DEFAULT_PARAMS[1])),
			beta2_(params.value("beta2", DEFAULT_PARAMS[2]))
	{
	}

};

template<typename T>
constexpr double AdamBase<T>::DEFAULT_PARAMS[];

template<typename T, typename Enable = void>
class Adam{};

template<typename T>
class Adam <T, typename std::enable_if<!nnqs::is_complex_type<T>::value>::type >
	: public AdamBase<T>
{
public:
	using typename Optimizer<T>::Vector;
	using typename Optimizer<T>::RealVector;

private:

	int t;
	Vector m_;
	Vector v_;

	using AdamBase<T>::alpha_;
	using AdamBase<T>::beta1_;
	using AdamBase<T>::beta2_;
public:
	using AdamBase<T>::DEFAULT_PARAMS;

	Adam(double alpha = DEFAULT_PARAMS[0], double beta1 = DEFAULT_PARAMS[1], double beta2 = DEFAULT_PARAMS[2])
		: AdamBase<T>(alpha, beta1, beta2)
	{
	}

	Adam(const nlohmann::json& params)
		: AdamBase<T>(params)
	{
	}

	Vector getUpdate(const Vector& grad) override
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

		double epsnorm = eps*sqrt(1.0-pow(beta2_,t));
		Vector denom = v_.unaryExpr([epsnorm](T x){ return sqrt(x)+epsnorm; });

		double alphat = alpha_*sqrt(1.0-pow(beta2_,t))/(1.0-pow(beta1_,t));

		return -alphat*m_.cwiseQuotient(denom);
	}
};

template<typename T>
class Adam <T, typename std::enable_if<nnqs::is_complex_type<T>::value>::type >
	: public Optimizer<T>
{
public:
	using RealT = typename nnqs::remove_complex<T>::type;
	using typename Optimizer<T>::Vector;
	using typename Optimizer<T>::RealVector;


private:
	Adam<RealT> adam_;

public:
	Adam(double alpha = 1e-3, double beta1 = 0.9, double beta2 = 0.999)
		: adam_(alpha, beta1, beta2)
	{
	}
	Adam(const nlohmann::json& params)
		: adam_(params)
	{
	}
	nlohmann::json params() const
	{
		return adam_.params();
	}

	Vector getUpdate(const Vector& grad) override
	{
		RealVector resReal = adam_.getUpdate(Eigen::Map<RealVector>((RealT*)grad.data(), 2*grad.rows(), 1));
		return Eigen::Map<Vector>((T*)resReal.data(), grad.rows(), 1);
	}

};
} //namespace nnqs
#endif//NNQS_OPTIMIZERS_ADAM_HPP
