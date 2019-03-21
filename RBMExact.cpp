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
//#include "Hamiltonians/TFIsing.hpp"

#include "Optimizers/SGD.hpp"

#include "States/RBMState.hpp"
#include "Samplers/HamiltonianSamplerPT.hpp"
#include "Samplers/SimpleSamplerPT.hpp"

#include "SRMatExact.hpp"
#include "SROptimizerCG.hpp"


using namespace nnqs;
using std::ios;

int main(int argc, char** argv)
{
	using namespace nnqs;

	std::random_device rd;
	std::default_random_engine re(rd());

	std::cout << std::setprecision(8);

	const int numChains = 16;

	const int N = 12;
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
	sscanf(argv[1], "%d", &alpha);

	double a, b;
	sscanf(argv[2], "%lf", &a);
	sscanf(argv[3], "%lf", &b);
	std::cout << "#a: " << a << ", b:" << b << std::endl;

	using Machine = RBM<ValT, true>;
	Machine qs(N, alpha*N);
	qs.initializeRandom(re);
	XYZNNN ham(N, a, b);
	//TFIsing ham(N, -1.0, -delta);

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

	//SwapSamplerPT<Machine, std::default_random_engine> ss(qs, numChains);
	SimpleSamplerPT<Machine, std::default_random_engine> ss(qs, numChains);
	ss.initializeRandomEngine();

	SRMatExact<Machine> srex(qs, ham);
	SRMatFree<Machine> srm(qs);

	const int dim = qs.getDim();

	for(int ll = 0; ll <=  1000; ll++)
	{
		double lambda = std::max(lmax*pow(decaying,ll), lmin);

		Vector g1;
		Vector res1;
		Vector del1;
		double e1;
		{
			srex.constructExact();

			e1 = srex.getEnergy();
			auto corrMat = srex.corrMat();
			corrMat += lambda*Matrix::Identity(dim, dim);

			Eigen::LLT<Eigen::MatrixXcd> llt(corrMat);

			g1 = srex.getF();
			res1 = llt.solve(g1);
			del1 = srex.deltaMean();
		}

		Vector g2;
		Vector res2;
		Vector del2;
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
			del2 = srm.deltaMean();
		}

		Vector optV = opt.getUpdate(res1);
		qs.updateParams(optV);

		ValT t = g1.adjoint()*g2;
		double x = (t.real() + t.imag())/(g1.norm()*g2.norm());

		std::cout << ll << "\t" << e1 << "\t" << e2 << "\t" <<
			g1.norm() << "\t" << g2.norm() << "\t" <<  x << std::endl;

	}

	return 0;
}
