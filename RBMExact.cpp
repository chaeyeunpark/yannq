#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <ios>
#include <boost/archive/binary_oarchive.hpp>
#include <nlohmann/json.hpp>

#include "Machines/RBM.hpp"
#include "Serializers/SerializeRBM.hpp"
#include "Hamiltonians/XYZNNN.hpp"
#include "Optimizers/SGD.hpp"

#include "SRMatExact.hpp"


using namespace nnqs;
using std::ios;

int main(int argc, char** argv)
{
	using namespace nnqs;

	
	std::random_device rd;
	std::default_random_engine re(rd());

	std::cout << std::setprecision(8);

	const int N = 16;
	const double decaying = 0.9;
	const double lmax = 10.0;
	const double lmin = 1e-3;
	const double sgd_eta = 0.05;

	using ValT = std::complex<double>;

	if(argc != 4)
	{
		printf("Usage: %s [alpha] [a] [b]\n", argv[0]);
		return 1;
	}
	int alpha;
	double a, b;
	sscanf(argv[1], "%d", &alpha);
	sscanf(argv[2], "%lf", &a);
	sscanf(argv[3], "%lf", &b);
	std::cout << "#a: " << a << ", b:" << b << std::endl;

	using Machine = RBM<ValT, false>;
	Machine qs(N, alpha*N);
	qs.initializeRandom(re);
	XYZNNN ham(N, a, b);

	const int dim = qs.getDim();

	SGD<ValT> opt(sgd_eta);

	{
		using nlohmann::json;
		json j;
		j["Optimizer"] = opt.params();
		j["Hamiltonian"] = ham.params();
		
		json lambda = 
		{
			{"decaying", decaying},
			{"lmax", lmax},
			{"lmin", lmin},
		};
		j["lambda"] = lambda;
		j["numThreads"] = Eigen::nbThreads();
		j["machine"] = qs.params();


		std::ofstream fout("params.dat");
		fout << j;
		fout.close();
	}

	typedef std::chrono::high_resolution_clock Clock;

	
	using std::sqrt;
	using Vector = typename Machine::Vector;
	using Matrix = typename Machine::Matrix;

	SRMatExact srSolver(qs, ham);

	for(int ll = 0; ll <=  7000; ll++)
	{
		if(ll % 10 == 0)
		{
			char fileName[30];
			sprintf(fileName, "w%04d.dat",ll);
			std::fstream out(fileName, ios::binary|ios::out);
			{
				boost::archive::binary_oarchive oa(out);
				oa << qs;
			}
		}
		
		double lambda = std::max(lmax*pow(decaying,ll), lmin);
		srSolver.constructExact();
		double currE = srSolver.getEnergy();
		auto corrMat = srSolver.corrMat();
		corrMat += lambda*Matrix::Identity(dim, dim);

		Eigen::LLT<MatrixXcd> llt(corrMat);
		Vector v = llt.solve(srm.getF());

		Vector optV = opt.getUpdate(v);

		double nv = v.norm();
		qs.updateParams(optV);

		std::cout << ll << "\t" << currE << "\t" << nv << << std::endl;
	}

	return 0;
}
