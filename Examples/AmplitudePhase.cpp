#include <iostream>
#include <nlohmann/json.hpp>

#include "Hamiltonians/XXXJ1J2.hpp"
#include "Machines/AmplitudePhase.hpp"
#include "Machines/layers/layers.hpp"
#include "Machines/FeedForward.hpp"

#include "Runners/RunAmplitudePhase.hpp"

int main(int argc, char* argv[])
{
	using namespace yannq;
	using namespace Eigen;

	using json = nlohmann::json;


	std::random_device rd;
	std::default_random_engine re{rd()};


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
	const double j2 = paramIn.at("j2").get<double>();

	std::cout << "#J2: " << j2 << std::endl;

	//construct feed forward network for phase
	FeedForward<double> ff;
	const int kernel_size = 5;
	ff.template addLayer<Conv1D>(1, 12, kernel_size, 1, false);
	ff.template addLayer<LeakyHardTanh>(0.01);
	ff.template addLayer<Conv1D>(12, 10, kernel_size, 1, false);
	ff.template addLayer<LeakyHardTanh>(0.01);
	ff.template addLayer<Conv1D>(10, 8, kernel_size, 1, false);
	ff.template addLayer<LeakyHardTanh>(0.01);
	ff.template addLayer<Conv1D>(8, 6, kernel_size, 1, false);
	ff.template addLayer<LeakyHardTanh>(0.01);
	ff.template addLayer<Conv1D>(6, 4, kernel_size, 1, false);
	ff.template addLayer<LeakyHardTanh>(0.01);
	ff.template addLayer<Conv1D>(4, 2, kernel_size, 1, false);
	ff.template addLayer<LeakyHardTanh>(0.01);
	ff.template addLayer<FullyConnected>(2*N, 1, false);
	ff.template addLayer<SoftSign>();

	std::cout << ff.summary() << std::endl;

	XXXJ1J2 ham(N, 1.0, j2, true);
	auto callback = [](int ll, double currE, double nGrad, double nv)
	{
		std::cout << ll << "\t" << currE << "\t" << nGrad << "\t" << nv << std::endl;
	};

	auto runner = RunAmplitudePhaseExact(N, alpha, false, std::move(ff), std::cerr);

	runner.initializeRandom(0.01);
	runner.setLambda(10.0, 0.9, 1e-3);
	runner.setIterParams(200, 0);
	runner.setOptimizer(paramIn["Optimizer"]);

	{
		json j = runner.getParams();
		j["Hamiltonian"] = ham.params();
		std::ofstream fout("paramOut.json");
		fout << j << std::endl;
	}
	auto basis = BasisJz{N,N/2};
	runner.run(callback, std::move(basis), std::move(ham));

	return 0;
}
