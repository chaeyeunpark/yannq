#include <iostream>
#include <iomanip>
#include <chrono>

#include "NNQS.hpp"
#include "SamplingResult.hpp"
#include "SimpleSampler.hpp"
#include "SROptimizer.hpp"
#include "XYZNNN.hpp"

using namespace yannq;
int main(int argc, char* argv[])
{
	using namespace yannq;

	const int N  = 12;
	const int alpha = 1;
	
	std::random_device rd;
	std::default_random_engine re(rd());

	std::cout << std::setprecision(8);
	
	using ValT = std::complex<double>;
	

	double weight = 0.05;
	if(argc != 1)
	{
		sscanf(argv[1], "%lf", &weight);
	}
	std::cout << "Weight: " << weight << std::endl;
	NNQS<ValT> qs(N, N);
	qs.initializeRandom(re, weight);

	double a = 1.0;
	double b = 0.6;

	std::cout << "a: " << a << "\t b: " << b << std::endl;

	XYZNNN ham(N, a, b);
	
	NNQS<ValT>::Vector psi = getPsi(qs);

	SimpleSampler<ValT> ss(qs);
	ss.setSigma(randomSigma(N,re));
	auto sr = ss.sampling(re, 1000);

	using std::norm;
	for(const auto& s: sr)
	{
		auto p = StateRef<ValT>(&qs, std::get<0>(s), std::get<1>(s));
		ValT res = ham(p);
		auto si = toValue(p.getSigma());
		auto res2 = ValT(ham.getCol(si).transpose()*psi)/psi.coeff(si);
		assert(std::norm(res - res2) < 1e-5);
	}
	return 0;
}
