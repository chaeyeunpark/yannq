#ifndef NNQS_OPTIMIZERS_ADADELTA_HPP
#define NNQS_OPTIMIZERS_ADADELTA_HPP
#include <Eigen/Dense>
#include <type_traits>
#include <complex>
#include "Utilities/type_traits.hpp"
#include "Optimizers/Optimizer.hpp"

namespace yannq
{

template<typename T>
class AdaDelta
	: public OptimizerGeometry<T>
{
public:
	using typename OptimizerGeometry<T>::RealT;
	using typename OptimizerGeometry<T>::Vector;
	using typename OptimizerGeometry<T>::RealVector;

private:
	const double rho_;

	int t;
	RealVector v_;
	RealVector x_;

public:
	static constexpr double DEFAULT_PARAMS[1] = {0.95};

	static nlohmann::json defaultParams()
	{
		return nlohmann::json
		{
			{"name", "AdaDelta"},
			{"rho", DEFAULT_PARAMS[0]},
		};
	}

	nlohmann::json params() const override
	{
		return nlohmann::json
		{
			{"name", "AdaDelta"},
			{"rho", rho_},
		};
	}

	AdaDelta(const nlohmann::json& params)
		: rho_(params.value("rho", DEFAULT_PARAMS[0])),
		t{}
	{
	}

	AdaDelta(double rho = DEFAULT_PARAMS[0])
		: rho_(rho)
	{
	}

	Vector getUpdate(const Vector& grad, const Vector& oloc) override
	{
		const double eps = 1e-6;
		auto toRMS = [eps](RealT x){ return sqrt(x)+eps; };
		auto norm = [](T x) -> RealT { return std::norm(x); };

		if(t == 0)
		{
			v_ = RealVector::Zero(grad.rows());
			x_ = RealVector::Zero(grad.rows());
		}
		++t;

		RealVector g2 = oloc.unaryExpr(norm);
		v_ *= (1-rho_);
		v_ += rho_*g2;
		
		RealVector num = x_.unaryExpr(toRMS);
		RealVector denom = v_.unaryExpr(toRMS);
		Vector res = -grad.cwiseProduct(num).cwiseQuotient(denom);

		x_ *= (1-rho_);
		x_ += rho_*res.unaryExpr(norm);

		return res;
	}

};
template<typename T>
constexpr double AdaDelta<T>::DEFAULT_PARAMS[];
} //namespace yannq
#endif//NNQS_OPTIMIZERS_ADADELTA_HPP
