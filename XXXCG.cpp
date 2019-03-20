#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>

#include "NNQS.hpp"
#include "SamplingResult.hpp"
#include "SimpleSampler.hpp"
#include "SROptimizerCG.hpp"

#include "XXX.hpp"

using namespace nnqs;

int main(int argc, char** argv)
{
	using namespace nnqs;

	const int N  = 28;
	
	std::random_device rd;
	std::default_random_engine re(rd());

	std::cout << std::setprecision(8);

	const double decaying = 0.9;
	double cliffThreshold = 30.0;
	double lmin = 1e-4;

	double lmax = 100.0;
	double eta_ini = 0.5;
	double eta_min = 1e-3;

	using ValT = std::complex<double>;

	if(argc != 4)
	{
		printf("Usage: %s [eta_ini] [eta_min] [lmax]\n", argv[0]);
		return 1;
	}
	sscanf(argv[1], "%lg", &eta_ini);
	sscanf(argv[2], "%lg", &eta_min);
	sscanf(argv[3], "%lg", &lmax);
	std::cout << "#eta_ini: " << eta_ini << ", eta_min: " << eta_min << ", lmax: " << lmax << std::endl;

	double alpha = 1.0;
	NNQS<ValT> qs(N, alpha*N);
	qs.initializeRandom(re);
	XXX ham(N);

	const int dim = qs.getDim();


	{
		std::ofstream fout("params.dat");
		fout << "{" << std::endl;
		fout << "\"N\": " << N << "," << std::endl;
		fout << "\"alpha\": " << alpha << "," << std::endl;
		fout << "\"eta_ini\": " << eta_ini << "," << std::endl;
		fout << "\"eta_min\": " << eta_min << "," << std::endl;
		fout << "\"lma\"x: " << lmax << ", " << std::endl;
		fout << "}" << std::endl;
		fout.close();
	}

	typedef std::chrono::high_resolution_clock Clock;

	SimpleSampler<ValT> ss(qs);
	SRMatFree<ValT> srm(&qs);

	using std::sqrt;

	for(int ll = 0; ll <=  5000; ll++)
	{
		if(ll % 10 == 0)
		{
			char fileName[30];
			sprintf(fileName, "w%04d.dat",ll);
			qs.save(fileName);
		}

		//Sampling
		auto smp_start = Clock::now();
		ss.setSigma(randomSigma(N,re));
		auto sr = ss.sampling(re, 10000);
		auto smp_dur = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - smp_start).count();

		auto slv_start = Clock::now();
		Eigen::ConjugateGradient<SRMatFree<ValT>, Eigen::Lower|Eigen::Upper, Eigen::IdentityPreconditioner> cg;
		srm.constructFromSampling(sr, ham);
		double lambda = std::max(lmax*pow(decaying,ll), lmin);
		double currE = srm.getEloc();
		srm.setShift(lambda);
		cg.compute(srm);
		cg.setTolerance(1e-3);
		typename NNQS<ValT>::Vector v = cg.solve(srm.getF());
		auto slv_dur = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - slv_start).count();

		double nv = v.norm();
		if(nv > cliffThreshold)
			v *= cliffThreshold/nv;

		double eta = std::max(eta_ini/(double(ll+1)), eta_min);
		qs.updateA(-eta*srm.getA(v));
		qs.updateB(-eta*srm.getB(v));
		qs.updateW(-eta*srm.getW(v));

		std::cout << ll << "\t" << currE << "\t" << nv << "\t" << eta << "\t" << smp_dur << "\t" << slv_dur << std::endl;
	}

	return 0;
}
