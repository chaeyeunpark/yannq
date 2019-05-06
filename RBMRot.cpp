#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <ios>
#include <boost/archive/binary_oarchive.hpp>
#include <nlohmann/json.hpp>

#include "Machines/RBM.hpp"
#include "States/RBMState.hpp"
#include "Samplers/SimpleSamplerPT.hpp"
#include "Serializers/SerializeRBM.hpp"
#include "Hamiltonians/XYZNNNRot.hpp"
#include "Optimizers/SGD.hpp"

#include "SROptimizerCG.hpp"


using namespace yannq;
using std::ios;

int main(int argc, char** argv)
{
	using namespace yannq;

	constexpr int numChains = 16;
	
	std::random_device rd;
	std::default_random_engine re(rd());

	std::cout << std::setprecision(8);

	const double decaying = 0.9;
	const double lmax = 10.0;
	const double lmin = 1e-3;
	const double sgd_eta = 0.05;

	using ValT = std::complex<double>;

	if(argc != 5)
	{
		printf("Usage: %s [n] [alpha] [a] [b]\n", argv[0]);
		return 1;
	}
	int N;
	int alpha;
	double a, b;
	sscanf(argv[1], "%d", &N);
	sscanf(argv[2], "%d", &alpha);
	sscanf(argv[3], "%lf", &a);
	sscanf(argv[4], "%lf", &b);
	std::cout << "#a: " << a << ", b:" << b << std::endl;

	using Machine = RBM<ValT, true>;
	Machine qs(N, alpha*N);
	qs.initializeRandom(re);
	XYZNNNRot ham(N, a, b);

	const int dim = qs.getDim();

	SGD<ValT> opt(sgd_eta);

	{
		using nlohmann::json;
		json j;
		j["Optimizer"] = opt.params();
		j["Hamiltonian"] = ham.params();
		
		json lambda = 
		{
			{"decaying", decaying},
			{"lmax", lmax},
			{"lmin", lmin},
		};
		j["lambda"] = lambda;
		j["numThreads"] = Eigen::nbThreads();
		j["machine"] = qs.params();


		std::ofstream fout("params.dat");
		fout << j;
		fout.close();
	}

	typedef std::chrono::high_resolution_clock Clock;

	SimpleSamplerPT<ValT, Machine, std::default_random_engine> ss(qs, numChains);
	SRMatFree<Machine> srm(qs);
	
	ss.initializeRandomEngine();

	using std::sqrt;
	using Vector = typename Machine::Vector;

	for(int ll = 0; ll <=  7000; ll++)
	{
		if(ll % 10 == 0)
		{
			char fileName[30];
			sprintf(fileName, "w%04d.dat",ll);
			std::fstream out(fileName, ios::binary|ios::out);
			{
				boost::archive::binary_oarchive oa(out);
				oa << qs;
			}
		}
		ss.randomizeSigma();

		//Sampling
		auto smp_start = Clock::now();
		auto sr = ss.sampling(dim, int(0.2*dim));
		auto smp_dur = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - smp_start).count();

		auto slv_start = Clock::now();
		Eigen::ConjugateGradient<SRMatFree<Machine>, Eigen::Lower|Eigen::Upper, Eigen::IdentityPreconditioner> cg;
		srm.constructFromSampling(sr, ham);
		double lambda = std::max(lmax*pow(decaying,ll), lmin);
		double currE = srm.getEloc();
		srm.setShift(lambda);
		cg.compute(srm);
		cg.setTolerance(1e-4);
		Vector v = cg.solve(srm.getF());
		Vector optV = opt.getUpdate(v);
		auto slv_dur = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - slv_start).count();

		double cgErr = (srm.apply(v)-srm.getF()).norm();
		double nv = v.norm();

		/*
		if(nv > clippThreshold)
			v *= clippThreshold/nv;
		double eta = std::max(eta_ini/sqrt(double(ll+1)), eta_min);
		*/
		qs.updateParams(optV);

		std::cout << ll << "\t" << currE << "\t" << nv << "\t" << cgErr
			<< "\t" << smp_dur << "\t" << slv_dur << std::endl;
	}

	return 0;
}
