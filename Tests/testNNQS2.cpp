#include <iostream>
#include <iomanip>
#include <chrono>

#include "NNQS.hpp"
#include "SamplingResult.hpp"
#include "SimpleSampler.hpp"
#include "SROptimizer.hpp"
#include "XYZNNN.hpp"

#include <SparseMat/ConstructMat.hpp>
#include <SparseMat/LocalSparseOperator.hpp>

using namespace nnqs;

template<typename ColFunc>
Eigen::VectorXd getCol(int dim, ColFunc&& t, int i)
{
	Eigen::VectorXd res = Eigen::VectorXd::Zero(dim);

	for(auto& rr : t.getCol(i))
	{
		res(rr.first) = rr.second;
	}
	return res;
}

arma::sp_mat getXYZ(double a, double b)
{
	arma::umat loc(2,8);
	loc(0,0) = 0;
	loc(1,0) = 0;

	loc(0,1) = 3;
	loc(1,1) = 0;

	loc(0,2) = 1;
	loc(1,2) = 1;

	loc(0,3) = 2;
	loc(1,3) = 1;

	loc(0,4) = 1;
	loc(1,4) = 2;

	loc(0,5) = 2;
	loc(1,5) = 2;

	loc(0,6) = 0;
	loc(1,6) = 3;

	loc(0,7) = 3;
	loc(1,7) = 3;

	arma::vec values{1.0, a-b, -1, a+b, a+b, -1, a-b, 1.0};

	return arma::sp_mat{loc, values, 4, 4, false, false};
}

template<class Ham>
Eigen::MatrixXd getHam(int dim, const Ham& ham)
{
	Eigen::MatrixXd res = Eigen::MatrixXd::Zero(dim,dim);
	for(int i = 0; i < dim; i++)
	{
		res.col(i) = ham.getCol(i);
	}
	return res;
}

int main(int argc, char* argv[])
{
	using namespace nnqs;

	const int N  = 12;
	const int alpha = 1;
	
	std::random_device rd;
	std::default_random_engine re(rd());

	std::cout << std::setprecision(8);
	
	using ValT = std::complex<double>;
	
	NNQS<ValT> qs(N, N);

	double weight = 0.05;
	if(argc != 1)
	{
		sscanf(argv[1], "%lf", &weight);
	}
	std::cout << "Weight: " << weight << std::endl;
	qs.initializeRandom(re, weight);

	double a, b;
	if(argc != 3)
	{
		std::cerr << "Usage: " << argv[0] << " [a] [b]" << std::endl;
		return 1;
	}
	sscanf(argv[1], "%lf", &a);
	sscanf(argv[2], "%lf", &b);

	std::cout << "a: " << a << "\t b: " << b << std::endl;

	XYZNNN ham(N, a, b);
	
	/*
	std::cout << ham.getCol(4).transpose() << std::endl;

	auto hamMat = getHam(1<<N, ham);
	std::cout << hamMat << std::endl;
	*/
	/*
	NNQS<ValT>::Vector psi = getPsi(qs);

	ValT res = 0;
	for(const auto& s: sr)
	{
		auto p = StateRef<ValT>(&qs, std::get<0>(s), std::get<1>(s));
		res += ham(p);
		auto si = toValue(p.getSigma())
		auto res2 = psi.at(si);
	}
	res /= sr.size();
	std::cout << res << std::endl;
	*/
	

		arma::sp_mat m1 = getXYZ(a,b);
		arma::sp_mat m2 = getXYZ(b,a);
		std::cout << arma::mat(m1) << std::endl;
		std::cout << arma::mat(m2) << std::endl;
		edp::LocalSparseOperator<double> lso(N, 2);
		for(int i = 0; i < N; i++)
		{
			lso.addTwoSiteTerm(std::make_pair(i, (i+1)%N), m1);
			lso.addTwoSiteTerm(std::make_pair(i, (i+2)%N), m2);
		}

	for(int i = 0; i < (1<<N); i++)
	{
		auto t1 = ham.getCol(i);
		auto t2 = getCol(1<<N, lso, i);
		assert((t1-t2).squaredNorm() < 1e-6);

	}

	return 0;
}
