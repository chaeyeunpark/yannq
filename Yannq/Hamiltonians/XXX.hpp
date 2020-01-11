#ifndef HAMILTONIANS_XXX_HPP
#define HAMILTONIANS_XXX_HPP
#include <Eigen/Eigen>

class XXX
{
private:
	int n_;

public:

	XXX(int n)
		: n_(n)
	{
	}
	
	template<class State>
	typename State::ValueType operator()(const State& smp) const
	{
		typename State::ValueType s = 0.0;
		//Nearest-neighbor
		for(int i = 0; i < n_; i++)
		{
			double yysign = smp.sigmaAt(i)*smp.sigmaAt((i+1)%n_);
			s += yysign; //zz
			s += (1.0-yysign)*smp.ratio(i, (i+1)%n_); //xx+yy
		}
		return s;
	}

	std::map<uint32_t, double> operator()(uint32_t col) const
	{
		std::map<uint32_t,double> res;
		for(int i = 0; i < n_; i++)
		{
			int b1 = (col >> i) & 1;
			int b2 = (col >> ((i+1)%n_)) & 1;
			int sgn = (1-2*b1)*(1-2*b2);
			long long int x = (1 << i) | (1 << ((i+1)%(n_)));
			res[col ^ x] += 1-sgn;
			res[col] += sgn;
		}
		return res;
	}
};
#endif//HAMILTONIANS_XXX_HPP
