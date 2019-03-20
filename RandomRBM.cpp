#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <ios>
#include <boost/archive/binary_oarchive.hpp>

#include "Machines/RBM.hpp"
#include "SRMatExact.hpp"


using namespace nnqs;
using std::ios;

template<typename RandomEngine>
Eigen::MatrixXcd Smat(RandomEngine& re, int n, int m, double w)
{
	std::uniform_real_distribution<> urd(-w, w);
	Eigen::MatrixXcd W(n,m);
	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < m; j++)
		{
			W(i,j) = std::complex<double>(urd(re), urd(re))
		}
	}


}

int main(int argc, char** argv)
{
	using namespace nnqs;

	constexpr int N  = 20;
	
	std::random_device rd;
	std::default_random_engine re(rd());

	std::cout << std::setprecision(8);

	using ValT = std::complex<double>;
	using Machine = nnqs::RBM<ValT, true>;

	Machine qs(N, 3*N);
	qs.initializeRandom(re);

	SRMatExact<Machine> srm(qs);
	srm.constructExact();
	Eigen::MatrixXcd m = srm.corrMat();
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es;
	es.compute(m, Eigen::EigenvaluesOnly);

	std::cout << es.eigenvalues().transpose() << std::endl;

	return 0;
}
