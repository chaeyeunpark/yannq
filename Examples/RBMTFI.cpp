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

using namespace yannq;
using std::ios;

int main(int argc, char** argv)
{
	using namespace yannq;
	using nlohmann::json;

	const int nTmp = 4;
	const int K = 8;

	std::random_device rd;
	std::default_random_engine re(rd());

	std::cout << std::setprecision(8);

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
	const bool useCG = paramIn.value("useCG", false);

	std::cout << "#h: " << h << std::endl;

	TFIsing ham(N, -1.0, -h);
	auto callback = [](int ll, double currE, double nv, double cgErr, auto smp_dur, auto slv_dur)
	{
		std::cout << ll << "\t" << currE << "\t" << nv << "\t" << cgErr
			<< "\t" << smp_dur << "\t" << slv_dur << std::endl;
	};

	auto runner = RunRBM<ValT>(N, alpha, true, std::cerr);
	runner.initializeRandom(0.01);
	runner.setIterParams(2000, 100);
	runner.setOptimizer(paramIn["Optimizer"]);
	runner.setSolverParams(useCG, 1e-3);

	{
		json j = runner.getParams();
		j["Hamiltonian"] = ham.params();
		std::ofstream fout("paramOut.json");
		fout << j << std::endl;
	}

	auto randomizer = [](auto& sampler)
	{
		sampler.randomizeSigma();
	};

	LocalSweeper sweeper{N};
	SamplerMT<RBM<ValT>, std::default_random_engine, RBMStateValue<ValT>, LocalSweeper>
		sampler(runner.getQs(), nTmp, K, sweeper);

	runner.run(sampler, callback, randomizer, std::move(ham), 2000, 200);
	return 0;
}
