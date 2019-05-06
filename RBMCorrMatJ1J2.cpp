#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <ios>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>

#include <nlohmann/json.hpp>

#include <Eigen/Eigenvalues> 

#include "Machines/RBM.hpp"
#include "States/RBMState.hpp"
#include "Samplers/SamplerPT.hpp"
#include "Samplers/SwapSweeper.hpp"
#include "Hamiltonians/XXXJ1J2.hpp"
#include "Serializers/SerializeRBM.hpp"

#include "ProcessCorrmat.hpp"

using namespace yannq;
using std::ios;


int main(int argc, char** argv)
{
	using namespace yannq;
	constexpr int numChains = 32;
	
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

	int n = params["Machine"]["n"];
	int m = params["Machine"]["m"];

	double J2 = params["Hamiltonian"]["J2"];
	
	XXXJ1J2 ham(n, 1.0, J2);
	Machine qs(n, m);

	//using Sampler = SimpleSamplerPT<Machine, std::default_random_engine>;
	
	using Sampler = SamplerPT<Machine, std::default_random_engine, RBMStateValue<Machine>, SwapSweeper>;
	SwapSweeper sweeper(n);
	Sampler ss(qs, numChains, sweeper);

	auto randomizer = [n](Sampler& ss)
	{
		//ss.randomizeSigma();
		ss.randomizeSigma(n/2);
	};

	CorrMatProcessor<Machine, XXXJ1J2, Sampler, decltype(randomizer)> cmp(qs, ham, ss, randomizer, true);
	cmp.processAll(dirPath);
	//processCorrmat(dirPath, qs, ham, ss, randomizer);
	
	return 0;
}