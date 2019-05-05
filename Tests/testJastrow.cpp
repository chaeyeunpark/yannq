#include <complex>
#include <random>
#include <iostream>

#include "Jastrow.hpp"
#include "JastrowSamples.hpp"
#include "Utility.hpp"


int main()
{
	const int N = 5;
	using namespace yannq;

	using ValT = std::complex<double>;

	yannq::Jastrow<ValT> qs(N);

	std::random_device rd;
	std::default_random_engine re{rd()};

	qs.initializeRandom(re);

	auto sigma = randomSigma(N, re);
	JastrowState<ValT> js(qs, sigma);
	std::cout << js.logRatio(3) << std::endl;
	ValT t1 = qs.calcTheta(sigma);
	sigma(3) *= -1;
	ValT t2 = qs.calcTheta(sigma);
	std::cout << t2 - t1 << std::endl;

	std::cout << qs.getA() << std::endl;
	std::cout << qs.getJ() << std::endl;
	std::cout << sigma.transpose() << std::endl;
	std::cout << t1 << std::endl;
	std::cout << t2 << std::endl;


	return 0;
}
