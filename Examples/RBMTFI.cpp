#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <ios>
#include <boost/archive/binary_oarchive.hpp>

#include <nlohmann/json.hpp>

#include "Machines/RBM.hpp"
#include "States/RBMStateMT.hpp"
#include "States/RBMState.hpp"
#include "Samplers/SamplerPT.hpp"
#include "Samplers/LocalSweeper.hpp"

#include "Optimizers/SGD.hpp"
#include "Serializers/SerializeRBM.hpp"
#include "Hamiltonians/TFIsing.hpp"

#include "GroundState/SRMat.hpp"


using namespace yannq;
using std::ios;

int main(int argc, char** argv)
{
	using namespace yannq;
	using nlohmann::json;

	constexpr int numChains = 16;
	
	std::random_device rd;
	std::default_random_engine re(rd());

	std::cout << std::setprecision(8);

	const double sgd_eta = 0.01;

	using ValT = std::complex<double>;

	if(argc != 2)
	{
		printf("Usage: %s [params.json]\n", argv[0]);
		return 1;
	}
	json paramIn;
	std::ifstream fin(argv[1]);
	fin >> paramIn;
	fin.close();

	const int N = paramIn.at("N").get<int>();
	const int alpha = paramIn.at("alpha").get<int>();
	const double h = paramIn.at("h").get<double>();
	const double lambda = paramIn.value("lambda", 1e-3);
	const bool useCG = paramIn.value("useCG", true);

	std::cout << "#h: " << h << std::endl;

	using Machine = RBM<ValT, true>;
	Machine qs(N, alpha*N);
	qs.initializeRandom(re);
	TFIsing ham(N, -1.0, -h);

	const int dim = qs.getDim();

	SGD<ValT> opt(sgd_eta, 0.0);

	{
		json j;
		j["Optimizer"] = opt.params();
		j["Hamiltonian"] = ham.params();
		
		j["lambda"] = lambda;
		j["numThreads"] = Eigen::nbThreads();
		j["machine"] = qs.params();
		j["useCG"] = useCG;


		std::ofstream fout("params.dat");
		fout << j;
		fout.close();
	}

	typedef std::chrono::high_resolution_clock Clock;

	LocalSweeper sweeper(N);
	SamplerPT<Machine, std::default_random_engine, RBMStateValue<Machine>, decltype(sweeper)> ss(qs, numChains, sweeper);
	SRMat<Machine, TFIsing> srm(qs, ham);
	
	ss.initializeRandomEngine();

	using std::sqrt;
	using Vector = typename Machine::Vector;

	for(int ll = 0; ll <=  2000; ll++)
	{
		if(ll % 100 == 0)
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
		auto samples = ss.sampling(dim, int(0.2*dim));
		auto smp_dur = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - smp_start).count();

		auto slv_start = Clock::now();
		
		srm.constructFromSampling(samples);

		double currE = std::real(srm.eloc());
		
		Vector v;
		if(useCG)
		{
			v = srm.solveCG(lambda);
		}
		else
		{
			v = srm.solveExact(lambda);
		}
		Vector optV = opt.getUpdate(v);
		auto slv_dur = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - slv_start).count();

		double cgErr = (srm.apply(v)-srm.energyGrad()).norm();
		double nv = v.norm();

		qs.updateParams(optV);

		std::cout << ll << "\t" << currE << "\t" << nv << "\t" << cgErr
			<< "\t" << smp_dur << "\t" << slv_dur << std::endl;
	}

	return 0;
}
