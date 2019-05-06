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
#include "Samplers/SamplerPT.hpp"
#include "Samplers/SwapSweeper.hpp"

#include "GroundState/SRMatExact.hpp"


using namespace yannq;
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
	using namespace yannq;
	using nlohmann::json;

	std::random_device rd;
	std::default_random_engine re(rd());

	std::cout << std::setprecision(8);

	const int numChains = 16;

	const double decaying = 0.9;
	const double lmax = 10.0;
	const double lmin = 1e-3;

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
	const bool useSR = paramIn.at("useSR").get<bool>();
	const bool printSv = paramIn.at("printSv").get<bool>();

	using Machine = RBM<ValT, true>;
	Machine qs(N, alpha*N);
	qs.initializeRandom(re);

	XXZ ham(N, 1.0, delta);
	auto opt = OptimizerFactory<ValT>::CreateOptimizer(paramIn.at("Optimizer")) ;

	{
		json j;
		j["Optimizer"] = opt->params();
		j["Hamiltonian"] = ham.params();
		
		json SR = 
		{
			{"useSR", useSR},
			{"decaying", decaying},
			{"lmax", lmax},
			{"lmin", lmin},
		};
		j["SR"] = SR;
		j["numThreads"] = Eigen::nbThreads();
		j["machine"] = qs.params();


		std::ofstream fout("params.dat");
		fout << j;
		fout.close();
	}

	using std::sqrt;
	using Vector = typename Machine::Vector;
	using Matrix = typename Machine::Matrix;

	auto basis = generateBasis(N, N/2);

	SwapSweeper sw(N);
	SamplerPT<Machine, std::default_random_engine, RBMStateValue<Machine>, SwapSweeper> ss(qs, numChains, sw);
	ss.initializeRandomEngine();

	SRMatExact<Machine> srex(qs, basis, ham);

	const int dim = qs.getDim();

	for(int ll = 0; ll <=  3000; ll++)
	{
		double lambda = std::max(lmax*pow(decaying,ll), lmin);

		if(ll % 50 == 0)
		{
			char fileName[30];
			sprintf(fileName, "w%04d.dat",ll);
			std::fstream out(fileName, ios::binary|ios::out);
			{
				boost::archive::binary_oarchive oa(out);
				oa << qs;
			}
		}


		srex.constructExact();

		double e = srex.getEnergy();
		auto corrMat = srex.corrMat();

		corrMat += lambda*Matrix::Identity(dim, dim);

		Eigen::LLT<Eigen::MatrixXcd> llt(corrMat);

		Vector v = llt.solve(srex.getF());
	
		Vector optV = opt->getUpdate(res1);
		qs.updateParams(optV);
		
		std::cout << ll << "\t" << e << std::endl;
	}

	return 0;
}
