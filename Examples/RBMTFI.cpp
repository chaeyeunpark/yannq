#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <ios>
#include <cereal/cereal.hpp>

#include <nlohmann/json.hpp>

#include <Samplers/SamplerMT.hpp>
#include <Runners/RunRBM.hpp>
#include <Hamiltonians/TFIsing.hpp>
#include <Utilities/Utility.hpp>

#include "tbb_threads.hpp"

using namespace yannq;
using std::ios;

int main(int argc, char** argv)
{
	using namespace yannq;
	using nlohmann::json;

	const int nTmp = 16;
	const int K = 1;

	std::random_device rd;
	std::default_random_engine re(rd());

	std::cout << std::setprecision(8);

	int nThreads = numThreads();
	tbb::global_control(tbb::global_control::max_allowed_parallelism, nThreads);
	std::cerr << "Using TBB nThreads: " << nThreads << std::endl;


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

	const uint32_t N = paramIn.at("N").get<int>();
	const int alpha = paramIn.at("alpha").get<int>();
	const double h = paramIn.at("h").get<double>();
	const bool useCG = paramIn.value("useCG", false);

	std::cout << "#h: " << h << std::endl;

	TFIsing ham(N, -1.0, -h);
	auto callback = [](int ll, double currE, double nv, double cgErr, auto smp_dur, auto slv_dur)
	{
		std::cout << ll << "\t" << currE << "\t" << nv << "\t" << cgErr
			<< "\t" << smp_dur << "\t" << slv_dur << std::endl;
	};

	RunRBM<ValT> runner(N, alpha, true, std::cerr);
	runner.initializeRandom(0.01);
	runner.setIterParams(20, 0);
	runner.setOptimizer(paramIn["Optimizer"]);
	runner.setSolverParams(useCG, 1e-3);

	{
		json j = runner.getParams();
		j["Hamiltonian"] = ham.params();
		std::ofstream fout("paramOut.json");
		fout << j << std::endl;
	}

	auto randomizer = [N](auto& re)
	{
		return randomSigma(N, re);
	};

	LocalSweeper sweeper{N};

	auto sampler = SamplerMT<RBM<ValT>, RBMStateValue<ValT>, LocalSweeper>
		(runner.getQs(), nTmp, K, sweeper);

	runner.run(sampler, callback, randomizer, std::move(ham), 2000, 200);
	return 0;
}
