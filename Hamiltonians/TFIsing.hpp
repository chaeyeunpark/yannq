#ifndef HAMILTONIANS_TFISING_HPP
#define HAMILTONIANS_TFISING_HPP
#include <Eigen/Eigen>
#include <nlohmann/json.hpp>

class TFIsing
{
private:
	int n_;
	double J_;
	double h_;
public:

	TFIsing(int n, double J, double h)
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
			s += J_*smp.sigmaAt(i)*smp.sigmaAt((i+1)%n_);
			s += h_*smp.ratio(i);
		}
		return s;
	}

	std::vector< std::array<int, 1> > offDiagonals(const Eigen::VectorXi& s) const
	{
		(void)s;
		std::vector< std::array<int, 1> > res;
		for(int i = 0; i < n_; i++)
		{
			res.push_back(std::array<int, 1>{i});
		}
		return res;
	}


	std::map<uint32_t, double> operator()(uint32_t col) const
	{
		std::map<uint32_t, double> res;
		for(int i = 0; i < n_; i++)
		{
			int s1 = (col >> i) & 1;
			int s2 = (col >> ((i+1) % n_)) & 1;
			long long int x = (1 << i);
			res[col ^ x] += h_;
			res[col] += J_*(1-2*s1)*(1-2*s2);
		}
		return res;
	}
};
#endif//HAMILTONIANS_TFISING_HPP
