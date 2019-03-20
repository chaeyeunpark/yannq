#ifndef CY_XYZ_NNN_ROT_HPP
#define CY_XYZ_NNN_ROT_HPP
#include <Eigen/Eigen>

class XYZNNNRot
{
private:
	int n_;
	double a_;
	double b_;
	constexpr static double J1 = -1.0;
	constexpr static double J2 = -1.0;
public:

	XYZNNNRot(int n, double a, double b)
		: n_(n), a_(a), b_(b)
	{
	}

	nlohmann::json params() const
	{
		return nlohmann::json
		{
			{"name", "XYZNNNRot"},
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
			s += -J1*b_*yysign; //zz
			s += J1*(a_*yysign+1.0)*smp.ratio(i, (i+1)%n_); //xx+yy
		}
		//Next-nearest-neighbor
		for(int i = 0; i < n_; i++)
		{
			double yysign = -smp.sigmaAt(i)*smp.sigmaAt((i+2)%n_);
			s += -J1*a_*yysign; //zz
			s += J1*(b_*yysign+1.0)*smp.ratio(i, (i+1)%n_); //xx+yy
		}
		return s;
	}

	std::vector< std::array<int, 2> > flips() const
	{
		std::vector< std::array<int, 2> > res;
		for(int i = 0; i < n_; i++)
		{
			res.push_back(std::array<int, 2>{i,(i+1)%n_});
			res.push_back(std::array<int, 2>{i,(i+2)%n_});
		}
		return res;
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
			res(col ^ x) += J1*(a_ - sgn*b_);
			res(col) += J1*sgn;
		}
		for(int i = 0; i < n_; i++)
		{
			int b1 = (col >> i) & 1;
			int b2 = (col >> ((i+2)%n_)) & 1;
			int sgn = (1-2*b1)*(1-2*b2);
			long long int x = (1 << i) | (1 << ((i+2)%(n_)));
			res(col ^ x) += J2*(b_ - sgn*a_);
			res(col) += J2*sgn;
		}
		return res;
	}
};
#endif//CY_XYZ_NNN_HPP
