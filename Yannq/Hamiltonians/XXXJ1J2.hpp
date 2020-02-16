#ifndef HAMILTONIANS_XXXXJ1J2_HPP
#define HAMILTONIANS_XXXXJ1J2_HPP
#include <Eigen/Eigen>
#include <nlohmann/json.hpp>

class XXXJ1J2
{
private:
	int n_;
	double J1_;
	double J2_;
	int sign_;

public:

	XXXJ1J2(int n, double J1, double J2, bool signRule)
		: n_(n), J1_(J1), J2_(J2)
	{
		if(signRule)
			sign_ = -1;
		else
			sign_ = 1;
	}

	nlohmann::json params() const
	{
		return nlohmann::json
		{
			{"name", "XXXXJ1J2"},
			{"n", n_},
			{"J1", J1_},
			{"J2", J2_}
		};
	}
	
	template<class State>
	typename State::T operator()(const State& smp) const
	{
		typename State::T s = 0.0;

		//Nearest-neighbor
		for(int i = 0; i < n_; i++)
		{
			double yysign = -smp.sigmaAt(i)*smp.sigmaAt((i+1)%n_);
			s += -J1_*yysign; //zz
			s += J1_*sign_*(1.0+yysign)*smp.ratio(i, (i+1)%n_); //xx+yy
		}
		//Next-nearest-neighbor
		for(int i = 0; i < n_; i++)
		{
			double yysign = -smp.sigmaAt(i)*smp.sigmaAt((i+2)%n_);
			s += -J2_*yysign; //zz
			s += J2_*(1.0+yysign)*smp.ratio(i, (i+2)%n_); //xx+yy
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
			m[col ^ x] += J1_*sign_*(1.0 - zz*1.0);
			m[col] += J1_*zz;
		}
		for(int i = 0; i < n_; i++)
		{
			int b1 = (col >> i) & 1;
			int b2 = (col >> ((i+2)%n_)) & 1;
			int zz = (1-2*b1)*(1-2*b2);
			long long int x = (1 << i) | (1 << ((i+2)%(n_)));
			m[col ^ x] += J2_*(1.0 - zz*1.0);
			m[col] += J2_*zz;
		}
		return m;
	}
};
#endif//HAMILTONIANS_XXXXJ1J2_HPP
