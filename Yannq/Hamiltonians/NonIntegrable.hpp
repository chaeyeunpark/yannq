#ifndef HAMILTONIANS_NONINTEGRABLE_HPP
#define HAMILTONIANS_NONINTEGRABLE_HPP
#include <Eigen/Eigen>
#include <nlohmann/json.hpp>

class NonIntegrable
{
private:
	int n_;
	double J_;
	double h_;
public:

	NonIntegrable(int n, double J, double h)
		: n_(n), J_(J), h_(h)
	{
	}

	nlohmann::json params() const
	{
		return nlohmann::json
		{
			{"name", "TFIsing"},
			{"n", n_},
			{"J", J_},
			{"h", h_}
		};
	}
	
	template<class State>
	typename State::T operator()(const State& smp) const
	{
		typename State::T s = 0.0;
		//Nearest-neighbor
		for(int i = 0; i < n_; i++)
		{
			s += -J_*smp.sigmaAt(i)*smp.sigmaAt((i+1)%n_);
			s += -1.0*smp.ratio(i);
			s += -h_*smp.sigmatAt(i);
		}
		return s;
	}

	Eigen::VectorXd getCol(long long int col) const
	{
		Eigen::VectorXd res = Eigen::VectorXd::Zero(1<<n_);
		for(int i = 0; i < n_; i++)
		{
			int s1 = (col >> i) & 1;
			int s2 = (col >> ((i+1) % n_)) & 1;
			long long int x = (1 << i);
			res(col ^ x) += -h_;
			res(col) += -J_*(1-2*s1)*(1-2*s2);
			res(col) += -1.0*(1-2*s1);
		}
		return res;
	}
};
#endif//HAMILTONIANS_NONINTEGRABLE_HPP
