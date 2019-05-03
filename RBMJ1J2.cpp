#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <ios>
#include <boost/archive/binary_oarchive.hpp>

#include <nlohmann/json.hpp>

#include "Machines/RBM.hpp"

#include "States/RBMState.hpp"
#include "Samplers/SamplerPT.hpp"
#include "Samplers/SwapSweeper.hpp"

#include "Serializers/SerializeRBM.hpp"

#include "Hamiltonians/XXXJ1J2.hpp"
#include "Hamiltonians/XXXMG.hpp"
#include "Optimizers/OptimizerFactory.hpp"

#include "SROptimizerCG.hpp"

using namespace nnqs;
using std::ios;

int main(int argc, char** argv)
{

	using namespace nnqs;
	using nlohmann::json;

	constexpr int numChains = 16;
	
	std::random_device rd;
	std::default_random_engine re(rd());

	std::cout << std::setprecision(8);

	const double decaying = 0.9;
	const double lmax = 10.0;
	const double lmin = 1e-4;

	const double cliffThreshold = 10.0;

	using ValT = std::complex<double>;
	using Machine = RBM<ValT, true>;
	using Hamiltonian = XXXMG;

	if(argc != 2)
	{
		printf("Usage: %s [param.json]\n", argv[0]);
		return 1;
	}

	json paramIn;
	std::ifstream fin(argv[1]);
	fin >> paramIn;
	fin.close();

	const int N = paramIn.at("N").get<int>();
	const int alpha = paramIn.at("alpha").get<int>();
	//const double J2 = paramIn.at("J2").get<double>();
	const bool useSR = paramIn.at("useSR").get<bool>();
	const bool printSv = paramIn.value("printSv", false);
	const bool useCliff = paramIn.value("useCliff", false);

	Machine qs{N, alpha*N};
	qs.initializeRandom(re);
	Hamiltonian ham{N};

	const int dim = qs.getDim();

	auto opt = OptimizerFactory<ValT>::getInstance().createOptimizer(paramIn.at("Optimizer")) ;

	{
		json j;
		j["printSv"] = printSv;
		j["Optimizer"] = opt->params();
		j["Hamiltonian"] = ham.params();
		
		json SR = 
		{
			{"useSR", useSR},
			{"decaying", decaying},
			{"lmax", lmax},
			{"lmin", lmin},
		};
		j["SR"] = SR;
		j["NumThreads"] = Eigen::nbThreads();
		j["Machine"] = qs.params();


		std::ofstream fout("params.dat");
		fout << j;
		fout.close();
	}

	typedef std::chrono::high_resolution_clock Clock;

	SwapSweeper sweeper(N);
	SamplerPT<Machine, std::default_random_engine, RBMStateValue<Machine>, SwapSweeper> ss(qs, numChains, sweeper);
	SRMatFree<Machine> srm(qs);
	
	ss.initializeRandomEngine();

	using std::sqrt;
	using Vector = typename Machine::Vector;

	for(int ll = 0; ll <=  3000; ll++)
	{
		if(ll % 5 == 0)
		{
			char fileName[30];
			sprintf(fileName, "w%04d.dat",ll);
			std::fstream out(fileName, ios::binary|ios::out);
			{
				boost::archive::binary_oarchive oa(out);
				oa << qs;
			}
		}
		ss.randomizeSigma(N/2);
		//ss.randomizeSigma();

		//Sampling
		auto smp_start = Clock::now();
		auto sr = ss.sampling(dim, int(0.2*dim));
		auto smp_dur = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - smp_start).count();

		auto slv_start = Clock::now();

		srm.constructFromSampling(sr, ham);
		double currE = srm.getEloc();
		double cgErr;
		
		Vector v;
		
		if(useSR)
		{
			Eigen::ConjugateGradient<SRMatFree<Machine>, Eigen::Lower|Eigen::Upper, Eigen::IdentityPreconditioner> cg;
			double lambda = std::max(lmax*pow(decaying,ll), lmin);
			srm.setShift(lambda);
			cg.compute(srm);
			cg.setTolerance(1e-4);
			v = cg.solve(srm.getF());
			cgErr = (srm.apply(v)-srm.getF()).norm();
		}
		else
		{
			v = srm.getF();
			cgErr = 0;
		}

		auto slv_dur = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - slv_start).count();

		if(printSv && ll % 5 == 0)
		{
			Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es;
			auto m = srm.corrMat();
			es.compute(m, Eigen::EigenvaluesOnly);

			char outputName[50];
			sprintf(outputName, "EV_W%04d.dat", ll);

			std::fstream out(outputName, ios::out);
			out << es.eigenvalues().transpose() << std::endl;
			out.close();
		}

		Vector optV = opt->getUpdate(v);

		if( useCliff && (optV.norm() > cliffThreshold))
		{
			optV /= optV.norm()*cliffThreshold;
		}

		double nv = v.norm();

		qs.updateParams(optV);
		
		std::cout << ll << "\t" << currE << "\t" << nv << "\t" << cgErr
			<< "\t" << smp_dur << "\t" << slv_dur << std::endl;

	}

	return 0;
}
