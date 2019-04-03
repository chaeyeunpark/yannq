#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <ios>
#include <boost/archive/binary_oarchive.hpp>
#include <nlohmann/json.hpp>

#include "Machines/Jastrow.hpp"
//#include "Machines/RBM.hpp"
//#include "Serializers/SerializeRBM.hpp"
//#include "Hamiltonians/XYZNNN.hpp"
#include "Hamiltonians/XXZ.hpp"
#include "Optimizers/SGD.hpp"

#include "States/JastrowState.hpp"
//#include "States/RBMState.hpp"
#include "Samplers/HamiltonianSampler.hpp"
//#include "Samplers/HamiltonianSamplerPT.hpp"
//#include "Samplers/SimpleSamplerPT.hpp"
//#include "Samplers/SimpleSampler.hpp"

#include "SRMatExact.hpp"
#include "SROptimizerCG.hpp"


using namespace nnqs;
using std::ios;

int main(int argc, char** argv)
{
	using namespace nnqs;

	const int numChains = 16;
	std::random_device rd;
	std::default_random_engine re(rd());

	std::cout << std::setprecision(8);

	const int N = 12;
	const double decaying = 0.9;
	const double lmax = 10.0;
	const double lmin = 1e-3;
	const double sgd_eta = 0.05;

	using ValT = std::complex<double>;

	if(argc != 2)
	{
		printf("Usage: %s [delta]\n", argv[0]);
		return 1;
	}
	double delta;
	sscanf(argv[1], "%lf", &delta);
	std::cout << "#delta: " << delta << std::endl;

	using Machine = Jastrow<ValT>;
	Machine qs(N);

	//using Machine = RBM<ValT, true>;
	//Machine qs(N, N);
	qs.initializeRandom(re);
	XXZ ham(N, 1.0, delta);

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

	//SimpleSampler<ValT, Machine, std::default_random_engine> ss(qs);
	HamiltonianSampler<Machine, std::default_random_engine, 2> ss(qs, ham.flips());
	ss.initializeRandomEngine();

	SRMatExact<Machine> srex(qs, ham);
	SRMatFree<Machine> srm(qs);

	const int dim = qs.getDim();

	for(int ll = 0; ll <= 2000; ll++)
	{

		double lambda = std::max(lmax*pow(decaying,ll), lmin);

		Vector g1;
		Vector res1;
		double e1;
		{
			srex.constructExact();

			e1 = srex.getEnergy();
			auto corrMat = srex.corrMat();
			corrMat += lambda*Matrix::Identity(dim, dim);

			Eigen::LLT<Eigen::MatrixXcd> llt(corrMat);

			g1 = srex.getF();
			res1 = llt.solve(g1);
		}

		Vector g2;
		Vector res2;
		double e2;
		{
			ss.randomizeSigma();
			auto sr = ss.sampling(2*dim, 0.4*dim);
			srm.constructFromSampling(sr, ham);
			
			Eigen::ConjugateGradient<SRMatFree<Machine>, Eigen::Lower|Eigen::Upper, Eigen::IdentityPreconditioner> cg;
			e2 = srm.getEloc();
			srm.setShift(lambda);
			cg.compute(srm);
			cg.setTolerance(1e-4);

			g2 = srm.getF();
			res2 = cg.solve(g2);
		}

		Vector optV = opt.getUpdate(res1);
		qs.updateParams(optV);

		std::cout << ll << "\t" << e1 << "\t" << e2 << "\t" <<
			(g1-g2).norm() << "\t" << (res1-res2).norm() << std::endl;
	}

	return 0;
}
