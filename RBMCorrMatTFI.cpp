#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <ios>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>

#include <nlohmann/json.hpp>

#include "Machines/RBM.hpp"
#include "States/RBMState.hpp"

#include "Samplers/LocalSweeper.hpp"
#include "Samplers/SamplerPT.hpp"
#include "Serializers/SerializeRBM.hpp"
#include "Hamiltonians/TFIsing.hpp"

#include "ProcessCorrmat.hpp"

using namespace yannq;
using std::ios;

int main(int argc, char** argv)
{
	using namespace yannq;
	constexpr int numChains = 32;
	
	std::random_device rd;
	std::default_random_engine re(rd());

	std::cout << std::setprecision(8);

	using ValT = std::complex<double>;
	using Machine = RBM<ValT, true>;

	using namespace boost::filesystem;
	using nlohmann::json;

	
	if(argc != 2 || !is_directory(argv[1]))
	{
		printf("Usage: %s [dir_path]\n", argv[0]);
		return 1;
	}
	path dirPath = argv[1];
	path paramPath = dirPath / "params.dat";
	
	json params;
	{
		ifstream fin(paramPath);
		fin >> params;
		fin.close();
	}

	ofstream fout("params.dat");
	fout << params;
	fout.close();

	int n = params["machine"]["n"];
	int m = params["machine"]["m"];
	double h = params["Hamiltonian"]["h"];

	std::vector<int> idxs;
	for(int i = 0; i <= 2000; i += 100)
	{
		idxs.emplace_back(i);
	}

	
	TFIsing ham(n, -1.0, h);
	Machine qs(n, m);
	LocalSweeper sweeper(n);
	using SamplerT=SamplerPT<Machine, std::default_random_engine, RBMStateValue<Machine>, decltype(sweeper)>;
	SamplerT ss(qs, numChains, sweeper);
	auto initSampler = [](SamplerT& ss)
	{
		ss.randomizeSigma();
	};

	CorrMatProcessor<Machine, TFIsing, SamplerT, decltype(initSampler)> cmp(qs, ham, ss, initSampler, true);
	cmp.processIdxs(dirPath, idxs);

	return 0;
}
