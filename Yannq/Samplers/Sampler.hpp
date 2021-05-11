#ifndef YANNQ_SAMPLER_SAMPLER_HPP
#define YANNQ_SAMPLER_SAMPLER_HPP
#include <random>
#include <memory>
#include <complex>

#include <nlohmann/json.hpp>
#include "Utilities/Utility.hpp"
namespace yannq
{
template<class Machine, class RandomEngine, class StateValue, class Sweeper>
class Sampler
{
private:
	uint32_t n_;
	const Machine& qs_;
	std::unique_ptr<StateValue> sv_;

	RandomEngine re_;
	Sweeper& sweeper_;

public:
	Sampler(const Machine& qs, Sweeper& sweeper)
		: n_(qs.getN()), qs_(qs), sweeper_(sweeper)
	{
	}

	nlohmann::json desc() const
	{
		nlohmann::json res;
		res["name"] = "Sampler";
		res["sweeper"] = sweeper_.name();
		return res;
	}

	void initializeRandomEngine()
	{
		std::random_device rd;
		re_.seed(rd());
	}

	/**
	 * @randomizer a function that output a random configuration. Must be thread safe.
	 * */
	void randomize(std::function<Eigen::VectorXi(RandomEngine&)> randomizer)
	{
		sv_ = std::make_unique<StateValue>(qs_, randomizer(re_.local()));
	}

	
	void randomizeSigma()
	{
		sv_ = std::make_unique<StateValue>(qs_, randomSigma(n_, re_));
	}
	void randomizeSigma(int nup)
	{
		sv_ = std::make_unique<StateValue>(qs_, randomSigma(n_, nup, re_));
	}

	inline void sweep()
	{
		sweeper_.sweep(*sv_, 1.0, re_);
	}

	auto sampling(int n_sweeps, int n_therm)
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
		using DataT = typename std::result_of<decltype(&StateValue::data)(StateValue)>::type;

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
} //yannq
#endif//YANNQ_SAMPLER_SAMPLER_HPP
