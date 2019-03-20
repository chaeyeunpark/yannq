#ifndef CY_LOCAL_SAMPLER_HPP
#define CY_LOCAL_SAMPLER_HPP
#include <cmath>
#include <Eigen/Eigen>
#include <memory>

#include "Samples.hpp"
#include "Utility.hpp"

namespace nnqs
{

template<typename T, class RandomEngine, class StateValueT = StateValue<T> >
class LocalSampler
{
private:
	int n_;
	NNQS<T>* qs_;
	std::unique_ptr<StateValueT> sv_;
	RandomEngine re_;

	std::uniform_int_distribution<> uid_;
	std::uniform_real_distribution<> urd_;

public:
	LocalSampler(NNQS<T>& qs)
		: n_(qs.getN()), qs_(&qs), sv_(nullptr), uid_(0, n_-1), urd_(0.0, 1.0)
	{
	}

	void setSigma(Eigen::VectorXi sigma)
	{
		using namespace std;
		sv_ = make_unique<StateValueT>(qs_, std::move(sigma));
	}

	void randomizeSigma()
	{
		sv_ = make_unique<StateValueT>(qs_, randomSigma(n_, re_));
	}

	void initializeRandomEngine()
	{
		std::random_device rd;
		re_.seed(rd());
	}
	void sweep()
	{
		for(int i = 0; i < n_; i++)
		{
			int toFlip = uid_(re_);
			double p = std::min(1.0,exp(2*real(sv_->logRatio(toFlip))));
			double u = urd_(re_);
			if(u < p)//accept
			{
				sv_->flip(toFlip);
			}
		}
	}


	auto sampling(int n_sweeps, int n_discard)
		-> std::vector<typename std::result_of<decltype(&StateValueT::data)(StateValueT)>::type>
	{

		using std::norm;
		using std::pow;
		using std::abs;
		using std::exp;


		//Thermalizing phase
		for(int n = 0; n < n_discard; n++)
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
#endif//CY_LOCAL_SAMPLER_HPP
