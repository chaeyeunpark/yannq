#ifndef HAMILTONIANS_XXZ_HPP
#define HAMILTONIANS_XXZ_HPP
#include <Eigen/Eigen>
#include <nlohmann/json.hpp>

class XXZ
{
private:
	int n_;
	double J_;
	double Delta_;
	double sign_;

public:

	XXZ(int n, double J, double Delta, bool signRule = false)
		: n_(n), J_(J), Delta_(Delta)
	{
		if(signRule)
			sign_ = 1.0;
		else
			sign_ = -1.0;
	}

	nlohmann::json params() const
	{
		return nlohmann::json
		{
			{"name", "XXZ"},
			{"n", n_},
			{"J", J_},
			{"Delta", Delta_},
			{"sign_rule", int(sign_)}
		};
	}
	
	template<class State>
	typename State::T operator()(const State& smp) const
	{
		typename State::T s = 0.0;
		//Nearest-neighbor
		for(int i = 0; i < n_; i++)
		{
			int zz = smp.sigmaAt(i)*smp.sigmaAt((i+1)%n_);
			s += J_*Delta_*zz; //zz
			s += sign_*J_*(1-zz)*smp.ratio(i, (i+1)%n_); //xx+yy
		}
		return s;
	}

	std::map<uint32_t, double> operator()(uint32_t col) const
	{
		std::map<uint32_t, double> m;
		for(int i = 0; i < n_; i++)
		{
			int b1 = (col >> i) & 1;
			int b2 = (col >> ((i+1)%n_)) & 1;
			int zz = (1-2*b1)*(1-2*b2);
			long long int x = (1 << i) | (1 << ((i+1)%(n_)));
			m[col] += J_*Delta_*zz;
			m[col ^ x] += sign_*J_*(1 - zz);
		}
		return m;
	}
};
#endif//HAMILTONIANS_XXZ_HPP
