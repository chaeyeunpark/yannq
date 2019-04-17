#ifndef NNQS_OPTIMIZERS_RMSPROP_HPP
#define NNQS_OPTIMIZERS_RMSPROP_HPP
#include <Eigen/Dense>
#include <type_traits>

#include <complex>
#include "Utilities/type_traits.hpp"
#include "Optimizers/Optimizer.hpp"

namespace nnqs
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

	int t;
	RealVector v_;

public:
	static constexpr double DEFAULT_PARAMS[] = {0.05, 0.9};

	static nlohmann::json defaultParams()
	{
		return nlohmann::json
		{
			{"name", "RMSProp"},
			{"alhpa", DEFAULT_PARAMS[0]},
			{"beta", DEFAULT_PARAMS[1]},
		};
	}

	nlohmann::json params() const override
	{
		return nlohmann::json
		{
			{"name", "RMSProp"},
			{"alhpa", alpha_},
			{"beta", beta_},
		};
	}

	RMSProp(const nlohmann::json& params)
		: alpha_(params.value("alpha", DEFAULT_PARAMS[0])), 
			beta_(params.value("beta", DEFAULT_PARAMS[1])),
			 t{}
	{
	}

	//use oloc to estimate local geometry 
	Vector getUpdate(const Vector& grad, const Vector& oloc) override
	{
		const double eps = 1e-8;
		auto sqr = [](T x) -> RealT {
			return std::norm(x);
		};

		if(t == 0)
		{
			v_ = RealVector::Zero(grad.rows());
		}
		++t;

		RealVector g = oloc.unaryExpr(sqr);
		v_ *= beta_;
		v_ += (1-beta_)*g;

		RealVector denom = v_.unaryExpr([eps](RealT x){ return sqrt(x)+eps; });
		return -alpha_*grad.cwiseQuotient(denom);
	}
};

template<typename T>
constexpr double RMSProp<T>::DEFAULT_PARAMS[];

} //namespace nnqs
#endif//NNQS_OPTIMIZERS_RMSPROP_HPP
