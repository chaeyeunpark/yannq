#ifndef NNQS_OPTIMIZERS_ADAMAX_HPP
#define NNQS_OPTIMIZERS_ADAMAX_HPP

#include "Optimizers/Optimizer.hpp"
namespace yannq
{
template<typename T>
class AdaMax
	: public OptimizerGeometry<T>
{
public:
	using Vector = typename OptimizerGeometry<T>::Vector;
	using RealVector = typename OptimizerGeometry<T>::RealVector;

	static constexpr double DEFAULT_PARAMS[] = {0.002, 0.9, 0.999};

private:
	double alpha_;
	double beta1_;
	double beta2_;
	
	int t_;

	Vector m_;
	RealVector u_;

public:

	AdaMax(double alpha = DEFAULT_PARAMS[0], double beta1 = DEFAULT_PARAMS[1], double beta2 = DEFAULT_PARAMS[2])
		: alpha_(alpha), beta1_(beta1), beta2_(beta2), t_{0}
	{
		u_ = 0;
	}

	AdaMax(const nlohmann::json& params)
		: alpha_(params.value("alpha", DEFAULT_PARAMS[0])), 
			beta1_(params.value("beta1", DEFAULT_PARAMS[1])),
			beta2_(params.value("beta2", DEFAULT_PARAMS[2])),
			t_{0}
	{
	}

	static nlohmann::json defaultParams()
	{
		return nlohmann::json
		{
			{"name", "AdaMax"},
			{"alhpa", DEFAULT_PARAMS[0]},
			{"beta1", DEFAULT_PARAMS[1]},
			{"beta2", DEFAULT_PARAMS[2]}
		};
	}

	nlohmann::json params() const override
	{
		return nlohmann::json
		{
			{"name", "AdaMax"},
			{"alhpa", alpha_},
			{"beta1", beta1_},
			{"beta2", beta2_}
		};

	}

	Vector getUpdate(const Vector& grad, const Vector& oloc) override
	{
		using std::pow;
		if(t_ == 0)
		{
			m_ = Vector::Zero(grad.rows());
			u_ = RealVector::Zero(grad.rows());
		}
		++t_;
		m_ *= beta1_;
		m_ += (1.0-beta1_)*grad;

		u_ *= beta2_;
		u_ = u_.cwiseMax(oloc.cwiseAbs());
		return -(alpha_/(1-pow(beta1_,t_)))*m_.cwiseQuotient(u_);
	}

};
}//namespace yannq
template<typename T>
constexpr double yannq::AdaMax<T>::DEFAULT_PARAMS[];
#endif//NNQS_OPTIMIZERS_ADAMAX_HPP
