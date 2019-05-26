#ifndef NNQS_SAMPLER_CD_HPP
#define NNQS_SAMPLER_CD_HPP
#include "Machines/RBM.hpp"
template<class Machinae, class CandidateGen>
class CD
{
	static_assert("Machine type must be RBM with real types");
};

template<bool useBias, class CandidateGen>
class CD<RBM<double, useBias>,  CandidateGen>
{
private:

	int k_;
	CandidateGen gen_;

public:
	CD(int k)
		: k_(k)
	{
	}

	sweep(const Eigen::VectorXi& v)
	{
		Vector theta = qs_.getW()*v;
		Vector p = 2*theta;
		p = p.exp();
#pragma omp parallel for schedule(static,8)
		for(int i = 0; i < p.size(); i++)
		{
			const auto t = p(i);
			p(i) = (t*t)/(1 + t*t);
		}
		Vector r(p.size());
		vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, r.data(), 0.0, 1.0);
		
	}
};
#endif//NNQS_SAMPLER_CD_HPP
