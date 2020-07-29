#ifndef NNQS_OPTIMIZERS_SGD_HPP
#define NNQS_OPTIMIZERS_SGD_HPP
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include "Optimizers/Optimizer.hpp"

namespace yannq
{
template<typename T>
class SGD
	: public Optimizer<T>
{
public:
	using typename Optimizer<T>::Vector;
	using typename Optimizer<T>::RealVector;

	static constexpr double DEFAULT_PARAMS[] = {0.05, 0.0};

private:
	double alpha_;
	double p_;
	int t_;

public:

	SGD(double alpha = DEFAULT_PARAMS[0], double p = DEFAULT_PARAMS[1])
		: alpha_{alpha}, p_{p}, t_{0}
	{
	}

	SGD(const nlohmann::json& params)
		: alpha_{params.value("alpha", DEFAULT_PARAMS[0])}, 
			p_{params.value("p", DEFAULT_PARAMS[1])},
			t_{0}
	{
	}

	nlohmann::json desc() const override
	{
		return nlohmann::json
		{
			{"name", "SGD"},
			{"alhpa", alpha_},
			{"p", p_}
		};
	}

	Vector getUpdate(const Vector& v) override
	{
		using std::pow;
		++t_;
		double eta = std::max((alpha_/pow(t_, p_)), 1e-4);
		return -eta*v;
	}
};
}//namespace yannq
template<typename T>
constexpr double yannq::SGD<T>::DEFAULT_PARAMS[];
#endif//NNQS_OPTIMIZERS_SGD_HPP
