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
#include "Samplers/SwapSamplerPT.hpp"
#include "Samplers/SimpleSamplerPT.hpp"
#include "Hamiltonians/XXZ.hpp"
#include "Serializers/SerializeRBM.hpp"

#include "ProcessCorrmat.hpp"

using namespace nnqs;
using std::ios;


int main(int argc, char** argv)
{
	using namespace nnqs;
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

	double delta = params["Hamiltonian"]["Delta"];
	
	XXZ ham(n, 1.0, delta);
	Machine qs(n, m);

	using Sampler = SimpleSamplerPT<Machine, std::default_random_engine>;
	//using Sampler = SwapSamplerPT<Machine, std::default_random_engine>;
	Sampler ss(qs, numChains);

	auto randomizer = [n](Sampler& ss)
	{
		ss.randomizeSigma();
		//ss.randomizeSigma(n/2);
	};
	processCorrmat(dirPath, qs, ham, ss, randomizer);
	
	return 0;
}
