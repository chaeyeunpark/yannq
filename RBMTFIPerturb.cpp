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
#include "Optimizers/SGD.hpp"
#include "Serializers/SerializeRBM.hpp"
#include "Hamiltonians/TFIPerturb.hpp"

#include "SROptimizerCG.hpp"


using namespace nnqs;
using std::ios;

int main(int argc, char** argv)
{
	using namespace nnqs;
	using nlohmann::json;

	constexpr int N  = 24;
	constexpr int numChains = 16;
	
	std::random_device rd;
	std::default_random_engine re(rd());

	std::cout << std::setprecision(8);

	const double decaying = 0.9;
	const double lmax = 10.0;
	const double lmin = 1e-3;

	//const double adam_eta = 0.05;
	const double sgd_eta = 0.05;

	using ValT = std::complex<double>;

	if(argc != 3)
	{
		printf("Usage: %s [alpha] [h]\n", argv[0]);
		return 1;
	}
	int alpha;
	double aa;
	sscanf(argv[1], "%d", &alpha);
	sscanf(argv[2], "%lf", &aa);
	std::cout << "#aa: " << aa << std::endl;

	using Machine = RBM<ValT, true>;
	Machine qs(N, alpha*N);
	qs.initializeRandom(re);
	TFIPerturb ham(N,aa);

	const int dim = qs.getDim();

	SGD<ValT> opt(sgd_eta);

	{
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

		qs.updateParams(optV);

		std::cout << ll << "\t" << currE << "\t" << nv << "\t" << cgErr
			<< "\t" << smp_dur << "\t" << slv_dur << std::endl;
	}

	return 0;
}
