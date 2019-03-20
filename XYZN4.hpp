#ifndef CY_XYZ_N4_HPP
#define CY_XYZ_N4_HPP
#include <Eigen/Eigen>

class XYZN4
{
private:
	int n_;
	double a_;
	double b_;
public:

	XYZN4(int n, double a, double b)
		: n_(n), a_(a), b_(b)
	{
	}
	
	template<class State>
	typename State::ValueType operator()(const State& smp) const
	{
		typename State::ValueType s = 0.0;
		//Nearest-neighbor
		for(int i = 0; i < n_; i++)
		{
			double yysign = -smp.sigmaAt(i)*smp.sigmaAt((i+1)%n_);
			s += -yysign; //zz
			s += (a_+yysign*b_)*smp.ratio(i, (i+1)%n_); //xx+yy
		}
		//Next-next-nearest-neighbor
		for(int i = 0; i < n_; i++)
		{
			double yysign = -smp.sigmaAt(i)*smp.sigmaAt((i+3)%n_);
			s += -yysign; //zz
			s += (b_+yysign*a_)*smp.ratio(i, (i+3)%n_); //xx+yy
		}
		return s;
	}

	Eigen::VectorXd getCol(long long int col) const
	{
		Eigen::VectorXd res = Eigen::VectorXd::Zero(1<<n_);
		for(int i = 0; i < n_; i++)
		{
			int b1 = (col >> i) & 1;
			int b2 = (col >> ((i+1)%n_)) & 1;
			int sgn = (1-2*b1)*(1-2*b2);
			long long int x = (1 << i) | (1 << ((i+1)%(n_)));
			res.coeffRef(col ^ x) += a_ - sgn*b_;
			res.coeffRef(col) += sgn;
		}
		for(int i = 0; i < n_; i++)
		{
			const int j = (i+3)%n_;
			int b1 = (col >> i) & 1;
			int b2 = (col >> j) & 1;
			int sgn = (1-2*b1)*(1-2*b2);
			long long int x = (1 << i) | (1 << j);
			res.coeffRef(col ^ x) += b_ - sgn*a_;
			res.coeffRef(col) += sgn;
		}
		return res;
	}
};
#endif//CY_XYZ_N4_HPP
