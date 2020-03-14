#include <Machines/RBM.hpp>
#include <Utilities/Utility.hpp>
int main()
{
	using namespace yannq;
	std::default_random_engine re;
	using ValT = std::complex<double>;
	RBM<ValT, true> rbm(20,60);
	re.seed(1234);
	rbm.initializeRandom(re, 1e-3);
	for(int n = 0; n < 1000; n++)
	{
		auto sigma = randomSigma(20, re);
		auto data = rbm.makeData(sigma);
		std::cout << rbm.logDeriv(data) << std::endl;
	}
	return 0;
}
