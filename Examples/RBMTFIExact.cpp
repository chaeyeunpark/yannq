#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <ios>
#include <cereal/cereal.hpp>

#include <nlohmann/json.hpp>

#include <Runners/RunRBMExact.hpp>
#include <Hamiltonians/TFIsing.hpp>

#include <Basis/BasisFull.hpp>

using namespace yannq;
using std::ios;

int main(int argc, char** argv)
{
	using namespace yannq;
	using nlohmann::json;

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

	std::cout << "#h: " << h << std::endl;

	TFIsing ham(N, -1.0, -h);
	auto callback = [](int ll, double currE, double nv)
	{
		std::cout << ll << "\t" << currE << "\t" << nv << std::endl;
	};

	auto runner = RunRBMExact<ValT>(N, alpha, true, std::cerr);
	runner.initializeRandom(0.01);
	runner.setIterParams(20, 0);
	runner.setOptimizer(paramIn["Optimizer"]);

	{
		json j = runner.getParams();
		j["Hamiltonian"] = ham.params();
		std::ofstream fout("paramOut.json");
		fout << j << std::endl;
	}

	runner.run(callback, BasisFull{N}, std::move(ham));
	return 0;
}
