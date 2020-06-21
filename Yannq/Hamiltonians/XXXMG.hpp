#ifndef HAMILTONIANS_XXXXMG_HPP
#define HAMILTONIANS_XXXXMG_HPP
#include <Eigen/Eigen>
#include <nlohmann/json.hpp>

class XXXMG
{
private:
	int n_;

public:

	XXXMG(int n)
		: n_(n)
	{
	}

	nlohmann::json params() const
	{
		return nlohmann::json
		{
			{"name", "XXXXMG"},
			{"n", n_},
		};
	}
	
	template<class State>
	typename State::Scalar operator()(const State& smp) const
	{
		constexpr double J1 = 1.0;
		constexpr double J2 = 0.5;
		typename State::Scalar s = 0.0;

		{
			double yysign = -smp.sigmaAt(0)*smp.sigmaAt(1);
			s += -J1/2*yysign; //zz
			s += J1/2*(1.0+yysign)*smp.ratio(0, 1); //xx+yy
		}
		{
			double yysign = -smp.sigmaAt(n_-2)*smp.sigmaAt(n_-1);
			s += -J1/2*yysign; //zz
			s += J1/2*(1.0+yysign)*smp.ratio(n_-2, n_-1); //xx+yy
		}	
		//Nearest-neighbor
		for(int i = 1; i < (n_-3); i++)
		{
			double yysign = -smp.sigmaAt(i)*smp.sigmaAt(i+1);
			s += -J1*yysign; //zz
			s += J1*(1.0+yysign)*smp.ratio(i, i+1); //xx+yy
		}
		//Next-nearest-neighbor
		for(int i = 0; i < n_-3; i++)
		{
			double yysign = -smp.sigmaAt(i)*smp.sigmaAt(i+2);
			s += -J2*yysign; //zz
			s += J2*(1.0+yysign)*smp.ratio(i, i+2); //xx+yy
		}
		return s;
	}

	std::map<uint32_t, double> operator()(uint32_t col) const
	{
		constexpr double J1 = 1.0;
		constexpr double J2 = 0.5;
		std::map<uint32_t, double> m;
		for(int i = 0; i < n_; i++)
		{
			int b1 = (col >> i) & 1;
			int b2 = (col >> ((i+1)%n_)) & 1;
			int sgn = (1-2*b1)*(1-2*b2);
			long long int x = (1 << i) | (1 << ((i+1)%(n_)));
			m[col ^ x] += J1*(1.0 - sgn*1.0);
			m[col] += J1*sgn;
		}
		for(int i = 0; i < n_; i++)
		{
			int b1 = (col >> i) & 1;
			int b2 = (col >> ((i+2)%n_)) & 1;
			int sgn = (1-2*b1)*(1-2*b2);
			long long int x = (1 << i) | (1 << ((i+2)%(n_)));
			m[col ^ x] += J2*(1.0 - sgn*1.0);
			m[col] += J2*sgn;
		}
		return m;
	}
};
#endif//HAMILTONIANS_XXXXMG_HPP
