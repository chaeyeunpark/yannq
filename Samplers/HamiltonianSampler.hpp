#ifndef CY_HAMILTONIAN_SAMPLER_HPP
#define CY_HAMILTONIAN_SAMPLER_HPP
#include <cmath>
#include <Eigen/Eigen>
#include <memory>
#include <random>
#include <iostream>

#include "Utilities/Utility.hpp"
#include "Utilities/type_traits.hpp"

namespace nnqs
{

template<class Machine, class Hamiltonian, class RandomEngine>
class HamiltonianSampler
{
public:
	using StateValueT = typename MachineStateTypes<Machine>::StateValue;
private:
	int n_;

	const Machine& qs_;
	const Hamiltonian& ham_;

	std::unique_ptr<StateValueT> sv_;
	RandomEngine re_;

public:
	HamiltonianSampler(Machine& qs, const Hamiltonian& ham)
		: n_(qs.getN()), qs_(qs), ham_(ham)
	{
	}

	void initializeRandomEngine()
	{
		std::random_device rd;
		re_.seed(rd());
	}
	
	void randomizeSigma()
	{
		sv_ = make_unique<StateValueT>(qs_, randomSigma(n_, re_));
	}
	

	void sweep()
	{
		using std::real;
		std::uniform_real_distribution<> urd(0.0,1.0);
		for(int sidx = 0; sidx < n_; sidx++)
		{
			auto& sigma = sv_->getSigma();
			auto flips = ham_.offDiagonals(sigma);

			std::uniform_int_distribution<> uid(0,flips.size()-1);
			auto toFlip = flips[uid(re_)];

			auto sigmap = sigma;
			for(int v: toFlip)
			{
				sigmap(v) *= -1;
			}
			const double w1 = flips.size();
			const double w2 = ham_.offDiagonals(sigmap).size();

			double p = std::min(1.0,exp(2.0*real(sv_->logRatio(toFlip)))*w1/w2);
			double u = urd(re_);

			if(u < p)//accept
			{
				sv_->flip(toFlip);
			}
		}
	}

	auto sampling(int n_sweeps, int n_therm)
		-> std::vector<typename std::result_of<decltype(&StateValueT::data)(StateValueT)>::type>
	{

		using std::norm;
		using std::pow;
		using std::abs;
		using std::exp;

		//Thermalizing phase
		for(int n = 0; n < n_therm; n++)
		{
			sweep();
		}
		using DataT = typename std::result_of<decltype(&StateValueT::data)(StateValueT)>::type;

		std::vector<DataT> res;
		res.reserve(n_sweeps);
		for(int ll = 0; ll < n_sweeps; ll++)
		{
			sweep();
			res.push_back(sv_->data());
		}
		return res;
	}
};
} //NNQS
#endif//CY_HAMILTONIAN_SAMPLER_HPP
