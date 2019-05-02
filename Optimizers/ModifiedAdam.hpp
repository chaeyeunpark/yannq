#ifndef NNQS_OPTIMIZERS_MODIFIEDADAM_HPP
#define NNQS_OPTIMIZERS_MODIFIEDADAM_HPP
#include <Eigen/Dense>
#include <type_traits>
#include "Utilities/type_traits.hpp"
#include "Optimizers/Optimizer.hpp"

namespace nnqs
{

template <typename T>
class ModifiedAdam
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
	Vector first_;
	RealVector second_;

public:
	static constexpr double DEFAULT_PARAMS[] = {1e-3, 0.95};

	static nlohmann::json defaultParams()
	{
		return nlohmann::json
		{
			{"name", "ModifiedAdam"},
			{"alhpa", DEFAULT_PARAMS[0]},
			{"beta", DEFAULT_PARAMS[1]},
		};
	}

	nlohmann::json params() const override
	{
		return nlohmann::json
		{
			{"name", "ModifiedAdam"},
			{"alhpa", alpha_},
			{"beta", beta_},
		};
	}

	ModifiedAdam(const nlohmann::json& params)
		: alpha_(params.value("alpha", DEFAULT_PARAMS[0])), 
			beta_(params.value("beta", DEFAULT_PARAMS[1])),
			t{}
	{
	}

	ModifiedAdam(double alpha = DEFAULT_PARAMS[0], double beta = DEFAULT_PARAMS[1])
		: alpha_(alpha), beta_(beta), t{}
	{
	}

	Vector getUpdate(const Vector& grad, const Vector& oloc) override
	{
		const double eps = 1e-5;
		auto norm = [](T x) -> RealT { return std::norm(x); };

		if(t ==0)
		{
			first_ = oloc;
			second_ = RealVector::Zero(oloc.rows());
		}
		else
		{
			Vector delta = oloc - first_;

			first_ += beta_*delta;

			RealVector g2 = delta.unaryExpr(norm);
			second_ *= (1-beta_);
			second_ += (1-beta_)*beta_*g2;
		}

		//RealVector denom = second_.unaryExpr([eps](RealT x)->RealT { return x + eps; });
		RealVector denom = second_.array() + std::max(1.0*pow(0.9,t),eps);
		std::cerr << denom.transpose() << std::endl;
		++t;

		return -alpha_*grad.cwiseQuotient(denom);
	}
};

template<typename T>
constexpr double ModifiedAdam<T>::DEFAULT_PARAMS[];

} //namespace nnqs
#endif//NNQS_OPTIMIZERS_MODIFIEDADAM_HPP
