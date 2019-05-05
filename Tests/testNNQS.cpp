#include "NNQS.hpp"
#include "SamplingResult.hpp"
#include "SimpleSampler.hpp"
#include "SROptimizer.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace yannq;

template<typename T>
class HamTI
{
private:
	int n_;
	double J_;
	double h_;
public:

	HamTI(int n, double J, double h)
		: n_(n), J_(J), h_(h)
	{
	}

	template<typename State>
	T operator()(const State& smp) const
	{
		T s = 0.0;
		for(int i = 0; i < n_; i++)
		{
			s += J_*smp.sigmaAt(i)*smp.sigmaAt((i+1)%n_);
			s += h_*smp.ratio(i);
		}
		return s;
	}


	typename NNQS<T>::Vector applyHam(const typename NNQS<T>::Vector& rhs)
	{
		typename NNQS<T>::Vector res = Eigen::VectorXd::Zero(rhs.size());
		for(int i = 0; i < rhs.size(); i++)
		{
			for(int k = 0; k < n_; k++)
			{
				int zz = ((i >> k) ^ (i >> ((k+1)%n_))) & 1; 
				double zz_d = 1-2*zz;
				res.coeffRef(i) += J_*rhs.coeff(i)*zz_d;
			}

			for(int k = 0; k < n_; k++)
			{
				res.coeffRef(i ^ (1<<k)) += h_*rhs.coeff(i);
			}
		}
		return res;
	}
};
int main(int argc, char* argv[])
{
	using namespace yannq;

	const int N  = 16;
	const int alpha = 1;
	
	const double J = -1.0;
	const double h = -0.5;
	
	std::random_device rd;
	std::default_random_engine re(rd());

	std::cout << std::setprecision(8);
	
	using ValT = std::complex<double>;
	
	NNQS<ValT> qs(N, N);

	double weight = 0.05;
	if(argc != 1)
	{
		sscanf(argv[1], "%lf", &weight);
	}
	std::cout << "Weight: " << weight << std::endl;
	qs.initializeRandom(re, weight);
	std::cout << qs.getW().squaredNorm() - static_cast<Eigen::MatrixXd>(qs.getW().real()).squaredNorm() << std::endl;

	HamTI<ValT> ham(N, J, h);

	{
		auto psi = getPsi(qs);
		std::cout << "#Initial E: " << (psi.adjoint()*(ham.applyHam(psi))).value() << std::endl;
	}

	SimpleSampler<ValT> ss(qs);
	ss.setSigma(randomSigma(N,re));
	auto sr = ss.sampling(re, 10000);

	ValT res = 0;
	for(const auto& s: sr)
	{
		auto p = StateRef<ValT>(&qs, std::get<0>(s), std::get<1>(s));
		res += ham(p);
	}
	res /= sr.size();
	std::cout << res << std::endl;


	return 0;
}
