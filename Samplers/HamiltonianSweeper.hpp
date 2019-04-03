#ifndef NNQS_HAMILTONIAN_SWEEPER_HPP
#define NNQS_HAMILTONIAN_SWEEPER_HPP
#include <random>
#include <complex>

namespace nnqs
{
template<class Hamiltonian>
class HamiltonianSweeper
{
private:
	int n_;
	const Hamiltonian& ham_;

public:
	HamiltonianSweeper(int n, const Hamiltonian& ham)
		: n_(n), ham_(ham)
	{
	}

	template<class StateValue, class RandomEngine>
	void localSweep(StateValue& sv, double beta, RandomEngine& re)
	{
		using std::real;
		std::uniform_real_distribution<> urd(0.0,1.0);
		for(int sidx = 0; sidx < n_; sidx++)
		{
			auto& sigma = sv.getSigma();
			auto flips = ham_.offDiagonals(sigma);

			std::uniform_int_distribution<> uid(0,flips.size()-1);
			auto toFlip = flips[uid(re)];

			auto sigmap = sigma;
			for(int v: toFlip)
			{
				sigmap(v) *= -1;
			}
			const double w1 = flips.size();
			const double w2 = ham_.offDiagonals(sigmap).size();

			double p = std::min(1.0,exp(2.0*real(sv.logRatio(toFlip)))*w1/w2);
			double u = urd(re);

			if(u < p)//accept
			{
				sv.flip(toFlip);
			}
		}
	}
};
} //NNQS
#endif//NNQS_HAMILTONIAN_SWEEPER_HPP
