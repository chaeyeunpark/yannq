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
#include "Hamiltonians/XXZ.hpp"
//#include "Hamiltonians/TFIsing.hpp"

#include "Optimizers/SGD.hpp"
#include "Optimizers/Adam.hpp"

#include "States/RBMState.hpp"
#include "Samplers/HamiltonianSamplerPT.hpp"
#include "Samplers/SimpleSamplerPT.hpp"
#include "Samplers/SwapSamplerPT.hpp"

#include "SRMatExactBasis.hpp"
#include "SROptimizerCG.hpp"


using namespace nnqs;
using std::ios;

std::vector<uint32_t> generateBasis(int n, int nup)
{
	std::vector<uint32_t> basis;
	uint32_t v = (1u<<nup)-1;
	uint32_t w;
	while(v < (1u<<n))
	{
		basis.emplace_back(v);
		uint32_t t = v | (v-1);
		w = (t + 1) | (((~t & -~t) - 1) >> (__builtin_ctz(v) + 1));
		v = w;
	}
	return basis;
}

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

	/*

	if(argc != 4)
	{
		printf("Usage: %s [alpha] [a] [b]\n", argv[0]);
		return 1;
	}
	*/

	int alpha;
	sscanf(argv[1], "%d", &alpha);

	/*
	double a, b;
	sscanf(argv[2], "%lf", &a);
	sscanf(argv[3], "%lf", &b);
	std::cout << "#a: " << a << ", b:" << b << std::endl;
	*/

	using Machine = RBM<ValT, true>;
	Machine qs(N, alpha*N);
	qs.initializeRandom(re);
	//XYZNNN ham(N, a, b);
	//TFIsing ham(N, -1.0, -delta);
	XXZ ham(N, 1.0, 1.1);
	Adam<ValT> opt{};


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

	auto basis = generateBasis(N,N/2);

	//SGD<ValT> opt(sgd_eta);

	SwapSamplerPT<Machine, std::default_random_engine> ss(qs, numChains);
	//SimpleSamplerPT<Machine, std::default_random_engine> ss(qs, numChains);
	ss.initializeRandomEngine();

	SRMatExactBasis<Machine> srex(qs, basis, ham);
	SRMatFree<Machine> srm(qs);

	const int dim = qs.getDim();

	for(int ll = 0; ll <=  3000; ll++)
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
			ss.randomizeSigma(N/2);
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

		Vector optV = opt.getUpdate(g1);
		qs.updateParams(optV);
		
		double t = g1.real().transpose()*g2.real();
		t += g1.imag().transpose()*g2.imag();
		double x = t/(g1.norm()*g2.norm());

		std::cout << ll << "\t" << e1 << "\t" << e2 << "\t" <<
			g1.norm() << "\t" << g2.norm() << "\t" <<  x << std::endl;

		/*
		std::cout << ll << "\t" << e1 << "\t" << g1.norm() << std::endl;
		*/

	}

	return 0;
}
