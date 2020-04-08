#ifndef YANNQ_SAMPLER_EXACTSAMPLER_HPP
#define YANNQ_SAMPLER_EXACTSAMPLER_HPP
#include <Machines/Machines.hpp>
#include <vector>
template<class Machine, class RandomEngine>
class ExactSampler
{
private:
	int n_;
	const Machine& qs_;
	std::vector<uint32_t> basis_;
	RandomEngine re_;

public:
	template<class Iterable>
	ExactSampler(const Machine& qs, Iterable&& basis)
		: n_(qs.getN()), qs_(qs)
	{
		basis_.reserve(basis.size());
		for(uint32_t v : basis)
			basis_.emplace_back(v);
	}

	void initializeRandomEngine()
	{
		std::random_device rd;
		re_.seed(rd());
	}
	
	auto sampling(uint32_t n_sweeps, uint32_t /*nSamplesDiscard*/)
	{
		using DataT = typename std::result_of_t<decltype(&Machine::makeData)(Machine, Eigen::VectorXi)>;

		auto st = getPsi(qs_, basis_, false);
		std::vector<double> accum(basis_.size()+1, 0.0);
		accum[0] = 0.0;
		for(uint32_t n = 1; n <= basis_.size(); n++)
		{
			accum[n] = accum[n-1] + std::norm(st(n-1));
		}

		std::uniform_real_distribution<> urd(0.0, accum[basis_.size()]);
		accum[basis_.size()] += 1e-8;
		tbb::concurrent_vector<DataT> res;
		tbb::queuing_mutex rMutex;
		res.reserve(n_sweeps);
		tbb::parallel_for(0u, n_sweeps,
			[&](uint32_t idx)
		{
			double r;
			{
				tbb::queuing_mutex::scoped_lock lock(rMutex);
				r = urd(re_);
			}
			auto iter = std::upper_bound(accum.begin(), accum.end(), r);
			auto idx = std::distance(accum.begin(), iter);
			auto data = qs_.makeData(toSigma(n_, basis_[idx]));
			res.emplace_back(data);
		}
		return res;
	}

};

#endif//YANNQ_SAMPLER_EXACTSAMPLER_HPP
