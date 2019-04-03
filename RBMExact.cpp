#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <ios>
#include <boost/archive/binary_oarchive.hpp>
#include <nlohmann/json.hpp>

#include "Machines/RBM.hpp"
#include "Serializers/SerializeRBM.hpp"

#include "Hamiltonians/XXZ.hpp"

#include "Optimizers/OptimizerFactory.hpp"

#include "States/RBMState.hpp"
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

template<typename T>
double cosBetween(const Eigen::Matrix<T, Eigen::Dynamic, 1>& g1, const Eigen::Matrix<T, Eigen::Dynamic, 1>& g2)
{
	double t = g1.real().transpose()*g2.real();
	t += g1.imag().transpose()*g2.imag();
	return t/(g1.norm()*g2.norm());
}

int main(int argc, char** argv)
{
	using namespace nnqs;
	using nlohmann::json;

	std::random_device rd;
	std::default_random_engine re(rd());

	std::cout << std::setprecision(8);

	const int numChains = 16;

	const double decaying = 0.9;
	const double lmax = 10.0;
	const double lmin = 1e-3;
	const double sgd_eta = 0.05;

	using ValT = std::complex<double>;

	if(argc != 2)
	{
		printf("Usage: %s [param.json]\n", argv[0]);
		return 1;
	}

	json paramIn;
	std::ifstream fin(argv[1]);
	fin >> paramIn;
	fin.close();

	const int N = paramIn.at("N").get<int>();
	const int alpha = paramIn.at("alpha").get<int>();
	const double delta = paramIn.at("delta").get<double>();

	using Machine = RBM<ValT, true>;
	Machine qs(N, alpha*N);
	qs.initializeRandom(re);

	XXZ ham(N, 1.0, delta);
	auto opt = OptimizerFactory<ValT>::CreateOptimizer(paramIn.at("Optimizer")) ;

	{
		json j;
		j["Optimizer"] = opt->params();
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

	auto basis = generateBasis(N, N/2);

	SwapSamplerPT<Machine, std::default_random_engine> ss(qs, numChains);
	ss.initializeRandomEngine();

	SRMatExactBasis<Machine> srex(qs, basis, ham);
	SRMatFree<Machine> srm(qs);

	const int dim = qs.getDim();

	for(int ll = 0; ll <=  3000; ll++)
	{
		double lambda = std::max(lmax*pow(decaying,ll), lmin);

		if(ll % 5 == 0)
		{
			char fileName[30];
			sprintf(fileName, "w%04d.dat",ll);
			std::fstream out(fileName, ios::binary|ios::out);
			{
				boost::archive::binary_oarchive oa(out);
				oa << qs;
			}
		}


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

			if(ll % 5 == 0)
			{
				Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es;
				es.compute(corrMat, Eigen::EigenvaluesOnly);

				char outputName[50];
				sprintf(outputName, "EV_W%04d.dat", ll);

				std::fstream out(outputName, ios::out);

				out << std::setprecision(16);
				out << es.eigenvalues().transpose() << std::endl;
				out.close();
			}
		}
	
		Vector g2;
		Vector res2;
		double e2;
		{
			ss.randomizeSigma(N/2);
			auto sr = ss.sampling(dim, 0.2*dim);
			srm.constructFromSampling(sr, ham);
			
			Eigen::ConjugateGradient<SRMatFree<Machine>, Eigen::Lower|Eigen::Upper, Eigen::IdentityPreconditioner> cg;
			e2 = srm.getEloc();
			srm.setShift(lambda);
			cg.compute(srm);
			cg.setTolerance(1e-4);

			g2 = srm.getF();
			res2 = cg.solve(g2);
		}

		Vector optV = opt->getUpdate(res1);
		qs.updateParams(optV);
		
		std::cout << ll << "\t" << e1 << "\t" << e2 << "\t" << g1.norm() << "\t" << g2.norm() << "\t" << 
			res1.norm() << "\t" << res2.norm() << "\t" <<
			(g1-g2).norm() << "\t" << (res1-res2).norm() << std::endl;
	}

	return 0;
}
