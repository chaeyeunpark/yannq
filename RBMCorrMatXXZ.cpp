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

#include "Samplers/HamiltonianSamplerPT.hpp"
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

	int n = params["machine"]["n"];
	int m = params["machine"]["m"];

	double delta = params["Hamiltonian"]["Delta"];
	
	XXZ ham(n, 1.0, delta);
	Machine qs(n, m);
	HamiltonianSamplerPT<Machine, std::default_random_engine,2> ss(qs, numChains, ham.flips());

	processCorrmat(dirPath, qs, ham, ss);
	
	return 0;
}
