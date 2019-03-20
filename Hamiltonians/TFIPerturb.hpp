#ifndef HAMILTONIANS_TFISING_HPP
#define HAMILTONIANS_TFISING_HPP
#include <Eigen/Eigen>
#include <nlohmann/json.hpp>

class TFIPerturb
{
private:
	int n_;
	double J_;
	double alpha_;
public:

	TFIPerturb(int n, double alpha)
		: n_(n), alpha_(alpha)
	{
		J_ = -1.0;
	}

	nlohmann::json params() const
	{
		return nlohmann::json
		{
			{"name", "TFIsing"},
			{"n", n_},
			{"J", J_},
			{"aa", alpha_}
		};
	}
	
	template<class State>
	typename State::T operator()(const State& smp) const
	{
		using T = typename State::T;
		T s = 0.0;
		constexpr T I{0.0,1.0};
		//Nearest-neighbor
		for(int i = 0; i < n_; i++)
		{
			int zz = smp.sigmaAt(i)*smp.sigmaAt((i+1)%n_);
			s += J_*zz;
			s += alpha_*I*smp.ratio(i)*T(smp.sigmaAt(i));
		}
		return s;
	}

	std::vector< std::array<int, 1> > flips() const
	{
		std::vector< std::array<int, 1> > res;
		for(int i = 0; i < n_; i++)
		{
			res.push_back(std::array<int, 1>{i});
		}
		return res;
	}


	Eigen::VectorXd getCol(long long int col) const
	{
		Eigen::VectorXd res = Eigen::VectorXd::Zero(1<<n_);
		for(int i = 0; i < n_; i++)
		{
			int s1 = (col >> i) & 1;
			int s2 = (col >> ((i+1) % n_)) & 1;
			long long int x = (1 << i);
			res(col) += J_*(1-2*s1)*(1-2*s2);
		}
		return res;
	}
};
#endif//HAMILTONIANS_TFISING_HPP
