#ifndef NNQS_OPTIMIZERS_ADAM_HPP
#define NNQS_OPTIMIZERS_ADAM_HPP
#include <Eigen/Dense>
#include <type_traits>
#include "Utilities/type_traits.hpp"
#include "Optimizers/Optimizer.hpp"

namespace yannq
{

template <typename T>
class Adam
	: public OptimizerGeometry<T>
{
public:
	using typename OptimizerGeometry<T>::RealT;
	using typename OptimizerGeometry<T>::Vector;
	using typename OptimizerGeometry<T>::RealVector;

private:
	const double alpha_;
	const double beta1_;
	const double beta2_;

	int t_;
	Vector m_;
	RealVector v_;

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

	Adam(const nlohmann::json& params)
		: alpha_(params.value("alpha", DEFAULT_PARAMS[0])), 
			beta1_(params.value("beta1", DEFAULT_PARAMS[1])),
			beta2_(params.value("beta2", DEFAULT_PARAMS[2])),
			t_{0}
	{
	}

	Adam(double alpha = DEFAULT_PARAMS[0], double beta1 = DEFAULT_PARAMS[1], double beta2 = DEFAULT_PARAMS[2])
		: alpha_(alpha), beta1_(beta1), beta2_(beta2), t_{0}
	{
	}

	Vector getUpdate(const Vector& grad, const Vector& oloc) override
	{
		const double eps = 1e-8;
		auto norm = [](T x) -> RealT { return std::norm(x); };

		if(t_ ==0)
		{
			m_ = Vector::Zero(grad.rows());
			v_ = RealVector::Zero(grad.rows());
		}
		++t_;

		m_ *= beta1_;
		m_ += (1-beta1_)*grad;

		RealVector g2 = oloc.unaryExpr(norm);
		v_ *= beta2_;
		v_ += (1-beta2_)*g2;

		double epsnorm = eps*sqrt(1.0-pow(beta2_,t_));
		RealVector denom = v_.unaryExpr([epsnorm](RealT x){ return sqrt(x)+epsnorm; });

		double alphat = alpha_*sqrt(1.0-pow(beta2_,t_))/(1.0-pow(beta1_,t_));

		return -alphat*m_.cwiseQuotient(denom);
	}
};

template<typename T>
constexpr double Adam<T>::DEFAULT_PARAMS[];

} //namespace yannq
#endif//NNQS_OPTIMIZERS_ADAM_HPP
