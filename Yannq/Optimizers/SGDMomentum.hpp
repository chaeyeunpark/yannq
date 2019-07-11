#ifndef NNQS_OPTIMIZERS_SGDMOMENTUM_HPP
#define NNQS_OPTIMIZERS_SGDMOMENTUM_HPP
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include "Optimizers/Optimizer.hpp"

namespace yannq
{
template<typename T>
class SGDMomentum
	: public Optimizer<T>
{
public:
	using typename Optimizer<T>::Vector;
	using typename Optimizer<T>::RealVector;

	static constexpr double DEFAULT_PARAMS[] = {0.05, 0.5, 0.9};

private:
	double alpha_;
	double p_;
	double gamma_;

	Vector m_;
	int t_;

public:

	SGDMomentum(double alpha = DEFAULT_PARAMS[0], double p = DEFAULT_PARAMS[1], double gamma = DEFAULT_PARAMS[2])
		: alpha_{alpha}, p_{p}, gamma_{gamma}, t_{0}
	{
	}

	SGDMomentum(const nlohmann::json& params)
		: alpha_{params.value("alpha", DEFAULT_PARAMS[0])}, 
			p_{params.value("p", DEFAULT_PARAMS[1])},
			gamma_{params.value("gamma", DEFAULT_PARAMS[2])},
			t_{0}
	{
	}

	nlohmann::json params() const override
	{
		return nlohmann::json
		{
			{"name", "SGD"},
			{"alhpa", alpha_},
			{"gamma", gamma_},
			{"p", p_}
		};
	}

	Vector getUpdate(const Vector& v) override
	{
		using std::pow;
		if(t_ == 0)
		{
			m_ = Vector::Zero(v.size());
		}

		++t_;
		m_ *= gamma_;
		m_ += v;
		double eta = std::max((alpha_/pow(t_, p_)), 1e-4);
		return -eta*m_;
	}
};
}//namespace yannq
template<typename T>
constexpr double yannq::SGDMomentum<T>::DEFAULT_PARAMS[];
#endif//NNQS_OPTIMIZERS_SGDMOMENTUM_HPP
