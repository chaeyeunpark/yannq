#ifndef HAMILTONIANS_XYZNNNSTO_HPP
#define HAMILTONIANS_XYZNNNSTO_HPP
#include <Eigen/Eigen>
#include <nlohmann/json.hpp>

/* Stoquastic for |a|, |b| < 1*/
class XYZNNNSto
{
private:
	int n_;
	double a_;
	double b_;
	constexpr static double J1 = -1.0;
	constexpr static double J2 = -1.0;
public:

	XYZNNNSto(int n, double a, double b)
		: n_(n), a_(a), b_(b)
	{
	}

	nlohmann::json params() const
	{
		return nlohmann::json
		{
			{"name", "XYZNNNSto"},
			{"n", n_},
			{"a", a_},
			{"b", b_}
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
			s += -J1*a_*yysign; //zz
			s += J1*(1.0+b_*yysign)*smp.ratio(i, (i+1)%n_); //xx+yy
		}
		//Next-nearest-neighbor
		for(int i = 0; i < n_; i++)
		{
			double yysign = -smp.sigmaAt(i)*smp.sigmaAt((i+2)%n_);
			s += -J2*b_*yysign; //zz
			s += J2*(1.0+a_*yysign)*smp.ratio(i, (i+2)%n_); //xx+yy
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
			int sgn = (1-2*b1)*(1-2*b2);
			long long int x = (1 << i) | (1 << ((i+1)%(n_)));
			m[col ^ x] += J1*(1.0 - sgn*b_);
			m[col] += J1*a_*sgn;
		}
		for(int i = 0; i < n_; i++)
		{
			int b1 = (col >> i) & 1;
			int b2 = (col >> ((i+2)%n_)) & 1;
			int sgn = (1-2*b1)*(1-2*b2);
			long long int x = (1 << i) | (1 << ((i+2)%(n_)));
			m[col ^ x] += J2*(1.0 - sgn*a_);
			m[col] += J2*b_*sgn;
		}
		return m;
	}
};
#endif//HAMILTONIANS_XYZNNNSTO_HPP
