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
#include "Samplers/SimpleSamplerPT.hpp"
#include "Serializers/SerializeRBM.hpp"
#include "Optimizers/SGD.hpp"

#include "SROptimizerCG.hpp"

#include "Hamiltonians/XYZNNN.hpp"
//#include "Hamiltonians/XXZ.hpp"

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
	path p = argv[1];
	p /= "params.dat";
	
	ifstream fin(p);
	
	json params;
	fin >> params;

	fin.close();

	ofstream fout("params.dat");
	fout << params;
	fout.close();

	int n = params["machine"]["n"];
	int m = params["machine"]["m"];

	double a = params["Hamiltonian"]["a"];
	double b = params["Hamiltonian"]["b"];
	//double delta = params["Hamiltonian"]["Delta"];
	
	XYZNNN ham(n, a, b);
	//XXZ ham(n, 1.0, delta);

	Machine qs(n, m);

	const int dim = qs.getDim();

	using std::sqrt;
	using Vector = typename Machine::Vector;

	std::vector<int> iters;
	for(int i = 0; i<=7000; i+=500)
	{
		iters.push_back(i);
	}

	char outName[] = "Energy.dat";
	std::fstream eDat(outName, ios::out);
	eDat << std::setprecision(10);

	for(auto ll : iters)
	{
		char fileName[30];
		sprintf(fileName, "w%04d.dat",ll);
		path file = argv[1];
		file /= fileName;
		std::cout << "Opening " << file << std::endl;
		fstream in(file, ios::binary|ios::in);
		{
			boost::archive::binary_iarchive ia(in);
			ia >> qs;
		}
		std::cout << "hasNaN?: " << qs.hasNaN() << std::endl;

		SimpleSamplerPT<ValT, Machine, std::default_random_engine> ss(qs, numChains);
		ss.initializeRandomEngine();
		ss.randomizeSigma();
		auto sr = ss.sampling(2*dim, int(0.2*2*dim));

		SRMatFree<Machine> srm(qs);
		srm.constructFromSampling(sr, ham);

		eDat << ll << "\t" << srm.getEloc() << "\t" << srm.getElocVar() << std::endl;

		auto m = srm.corrMat();

		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es;
		es.compute(m, Eigen::EigenvaluesOnly);

		char outputName[50];
		sprintf(outputName, "EV_W%04d.dat", ll);

		std::fstream out(outputName, ios::out);
		out << es.eigenvalues().transpose() << std::endl;
		out.close();
	}
	eDat.close();

	return 0;
}
