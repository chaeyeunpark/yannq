#ifndef NNQS_OPTIMIZERS_RMSPROP_HPP
#define NNQS_OPTIMIZERS_RMSPROP_HPP
#include <Eigen/Dense>
#include <type_traits>

#include <complex>
#include "Utilities/type_traits.hpp"
#include "Optimizers/Optimizer.hpp"

namespace yannq
{
template <typename T>
class RMSProp
	: public OptimizerGeometry<T>
{
public:
	using typename OptimizerGeometry<T>::RealT;
	using typename OptimizerGeometry<T>::Vector;
	using typename OptimizerGeometry<T>::RealVector;

private:
	const double alpha_;
	const double beta_;
	const double eps_;

	int t_;
	RealVector v_;

public:
	static constexpr double DEFAULT_PARAMS[] = {0.05, 0.9, 1e-8};

	static nlohmann::json defaultParams()
	{
		return nlohmann::json
		{
			{"name", "RMSProp"},
			{"alhpa", DEFAULT_PARAMS[0]},
			{"beta", DEFAULT_PARAMS[1]},
			{"eps", DEFAULT_PARAMS[2]},
		};
	}

	nlohmann::json params() const override
	{
		return nlohmann::json
		{
			{"name", "RMSProp"},
			{"alhpa", alpha_},
			{"beta", beta_},
			{"eps", eps_},
		};
	}

	RMSProp(const nlohmann::json& params)
		: alpha_(params.value("alpha", DEFAULT_PARAMS[0])), 
			beta_(params.value("beta", DEFAULT_PARAMS[1])),
			eps_(params.value("eps", DEFAULT_PARAMS[2])),
			 t_{}
	{
	}

	//use oloc to estimate local geometry 
	Vector getUpdate(const Vector& grad, const Vector& oloc) override
	{
		auto sqr = [](T x) -> RealT {
			return std::norm(x);
		};

		if(t_ == 0)
		{
			v_ = RealVector::Zero(grad.rows());
		}
		++t_;

		RealVector g = oloc.unaryExpr(sqr);
		v_ *= beta_;
		v_ += (1-beta_)*g;

		RealVector denom = v_.unaryExpr([eps = this->eps_](RealT x){ return sqrt(x)+eps; });
		return -alpha_*grad.cwiseQuotient(denom);
	}
};

template<typename T>
constexpr double RMSProp<T>::DEFAULT_PARAMS[];

} //namespace yannq
#endif//NNQS_OPTIMIZERS_RMSPROP_HPP
