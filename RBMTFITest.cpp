#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <ios>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <nlohmann/json.hpp>

#include "Machines/RBM.hpp"
#include "States/RBMStateMT.hpp"
#include "States/RBMState.hpp"
#include "Samplers/Sampler.hpp"
#include "Samplers/LocalSweeper.hpp"

#include "Optimizers/OptimizerFactory.hpp"
#include "Serializers/SerializeRBM.hpp"
#include "Hamiltonians/TFIsing.hpp"

#include "SROptimizerCG.hpp"

#include "Serializers/SerializeEigen.hpp"


using namespace yannq;
using std::ios;

int main(int argc, char** argv)
{
	using namespace yannq;
	using nlohmann::json;

	constexpr int N = 20;
	
	std::random_device rd;
	std::default_random_engine re(rd());

	std::cout << std::setprecision(8);

	using ValT = std::complex<double>;

	constexpr int alpha = 3;
	constexpr double h = 1.0;
	constexpr double lambda = 1e-3;

	using Machine = RBM<ValT, true>;
	Machine qs(N, alpha*N);
	
	TFIsing ham(N, -1.0, -h);

	if(argc != 2)
	{
		printf("Usage: %s [param.json]\n", argv[0]);
		return 1;
	}

	json paramIn;
	std::ifstream fin(argv[1]);
	fin >> paramIn;
	fin.close();

	const bool useCG = paramIn.at("useCG").get<bool>();
	auto opt = OptimizerFactory<ValT>::getInstance().createOptimizer(paramIn.at("Optimizer")) ;

	const std::string method = paramIn.at("RBM").at("random").get<std::string>();
	const double weight = paramIn.at("RBM").at("weight").get<double>();

	if(method.compare("uniform") == 0)
	{
		qs.initializeRandomUniform(re, weight);
	}
	else if(method.compare("normal") == 0)
	{
		qs.initializeRandom(re, weight);
	}
	else
	{
		std::cerr << "json error" << std::endl;
		return 1;
	}

	const int dim = qs.getDim();

	{
		json j;
		j["Optimizer"] = opt->params();
		j["Hamiltonian"] = ham.params();
		
		j["lambda"] = lambda;
		j["numThreads"] = Eigen::nbThreads();
		j["machine"] = qs.params();


		std::ofstream fout("params.dat");
		fout << j;
		fout.close();
	}

	typedef std::chrono::high_resolution_clock Clock;

	LocalSweeper sweeper(N);
	Sampler<Machine, std::default_random_engine, RBMStateValue<Machine>, decltype(sweeper)> ss(qs, sweeper);
	SRMatFree<Machine> srm(qs);
	
	ss.initializeRandomEngine();

	using std::sqrt;
	using Vector = typename Machine::Vector;

	for(int ll = 0; ll <=  2000; ll++)
	{
		if(ll % 100 == 0)
		{
			char fileName[30];
			sprintf(fileName, "w%04d.dat",ll);
			std::fstream out(fileName, ios::out);
			{
				boost::archive::text_oarchive oa(out);
				oa << qs;
			}
		}
		ss.randomizeSigma();

		//Sampling
		auto smp_start = Clock::now();
		auto sr = ss.sampling(dim, int(0.2*dim));
		auto smp_dur = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - smp_start).count();

		auto slv_start = Clock::now();
		srm.constructFromSampling(sr, ham);
		double currE = srm.getEloc();
		srm.setShift(lambda);

		Vector v;
		if(useCG){
			Eigen::ConjugateGradient<SRMatFree<Machine>, Eigen::Lower|Eigen::Upper, Eigen::IdentityPreconditioner> cg;
			cg.compute(srm);
			cg.setTolerance(1e-3);
			v = cg.solve(srm.getF());
		}
		else
		{
			using namespace Eigen;
			MatrixXcd A = srm.corrMat() + lambda*MatrixXcd::Identity(dim, dim);
			LLT<MatrixXcd> llt(A);
			v = llt.solve(srm.getF());
		}

		Vector optV = opt->getUpdate(v);
		auto slv_dur = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - slv_start).count();

		double cgErr = (srm.apply(v)-srm.getF()).norm();
		double nv = v.norm();

		qs.updateParams(optV);

		std::cout << ll << "\t" << currE << "\t" << nv << "\t" << cgErr
			<< "\t" << smp_dur << "\t" << slv_dur << std::endl;
	}

	return 0;
}
