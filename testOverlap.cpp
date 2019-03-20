#include <complex>
#include "NNQS.hpp"
#include "SimpleSampler.hpp"

#include <iostream>

int main()
{
	using namespace nnqs;
	const int N = 16;
	const int M = 16;

	typedef std::complex<double> cx_double;
	const cx_double I(0.0,1.0);

	std::random_device rd;
	std::default_random_engine re(rd());


	NNQS<cx_double> qs1(N,M);
	qs1.initializeRandom(re, 0.1);
	/*
	qs1.setW(Eigen::MatrixXcd::Zero(M,N));
	qs1.setA(Eigen::VectorXcd::Zero(N));
	qs1.setB(Eigen::VectorXcd::Zero(M));
	*/
	NNQS<cx_double> qs2(N,M);
	qs2.initializeRandom(re, 0.1);
	Eigen::VectorXcd ap = Eigen::VectorXcd::Zero(N);
	ap.coeffRef(0) = I*M_PI/2.0;
	//qs2.updateA(ap);

	std::cout << std::norm(inner(qs1, qs2)) << std::endl;

	SimpleSampler<cx_double> ss1(qs1, re);
	SimpleSampler<cx_double> ss2(qs2, re);
	SamplingResult<cx_double> t1 = ss1.sampling(re, 3000);
	SamplingResult<cx_double> t2 = ss2.sampling(re, 3000);

	cx_double s = 0;
	for(int i = 0; i < t1.size(); i++)
	{
		auto s1 = t1.getObjAt(i);
		for(int j = 0; j < t2.size(); j++)
		{
			auto s2 = t2.getObjAt(j);
			s += s1.ratio(s2.getSigma())*s2.ratio(s1.getSigma());
		}
	}
	s /= (t1.size())*(t2.size());
	std::cout << s << std::endl;

	return 0;
}
