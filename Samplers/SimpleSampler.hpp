#ifndef CY_SAMPLER_SIMPLESAMPLER_HPP
#define CY_SAMPLER_SIMPLESAMPLER_HPP
#include <cmath>
#include <Eigen/Eigen>
#include <memory>
#include <omp.h>

#include "Utilities/Utility.hpp"
#include "Utilities/type_traits.hpp"

namespace nnqs
{

template<typename T, class Machine, class RandomEngine>
class SimpleSampler
{
public:
	using StateValueT = typename MachineStateTypes<Machine>::StateValue;
private:
	int n_;
	int numChain_;
	Machine& qs_;
	std::unique_ptr<StateValueT> sv_;

	RandomEngine re_;

public:

	SimpleSampler(Machine& qs)
		: n_(qs.getN()), qs_(qs)
	{
		static_assert(std::is_same<T, typename Machine::ScalarType>::value, "T must be same to Machine scalar type");
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
		std::uniform_int_distribution<int> uid(0, n_-1);
		std::uniform_real_distribution<double> urd(0, 1.0);
		for(int sidx = 0; sidx < n_; sidx++)
		{
			int toFlip = uid(re_);
			double p = std::min(1.0, 2.0*real(sv_->logRatio(toFlip)));
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
#endif//CY_SAMPLER_SIMPLESAMPLER_HPP
