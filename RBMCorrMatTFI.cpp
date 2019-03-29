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

#include "Samplers/SimpleSamplerPT.hpp"
#include "Serializers/SerializeRBM.hpp"
#include "Hamiltonians/TFIsing.hpp"

#include "ProcessCorrmat.hpp"

using namespace nnqs;
using std::ios;

int main(int argc, char** argv)
{
	using namespace nnqs;
	constexpr int numChains = 32;
	
	std::random_device rd;
	std::default_random_engine re(rd());

	std::cout << std::setprecision(8);

	using ValT = std::complex<double>;
	using Machine = RBM<ValT, false>;

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
	
	TFIsing ham(n, -1.0, h);
	Machine qs(n, m);
	SimpleSamplerPT<Machine, std::default_random_engine> ss(qs, numChains);

	processCorrmat(dirPath, qs, ham, ss);

	return 0;
}
