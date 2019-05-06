#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <boost/archive/binary_oarchive.hpp>
#include <ios>

#include "Machines/RBM.hpp"
#include "Optimizers/Adam.hpp"
#include "Serializers/SerializeRBM.hpp"

#include "OverlapOptimizer.hpp"

Eigen::VectorXd apply(int n, const Eigen::VectorXd& vec, const Eigen::Matrix4d& unitary, int i1, int i2)
{
	Eigen::VectorXd res(1<<n);
	res.setZero();
	for(int i = 0; i < (1<<(n-2)); i++)
	{
		for(int j = 0; j < 4; j++)
		{
			unsigned int index = (i << 2) | j;
			bitswap(index, 1, i1);
			bitswap(index, 0, i2);
			cx_vec c = unitary.col(j);
			for(int k = 0; k < 4; k++)
			{
				unsigned int index_to = (i << 2) | k;
				bitswap(index_to, 1, i1);
				bitswap(index_to, 0, i2);
				res[index_to] += c[k]*vec[index];
			}
		}
	}
	return res;
}

class GraphState
{
private:
	int n_;

public:
	GraphState(int n)
		: n_ (n)
	{
	}


	Eigen::VectorXd generateState(const MatrixXi& adjMat)
	{
		Eigen::VectorXd v(1<<n_);
		v.setOnes();
		v /= std::pow(2,double(n_)/2);
		for(int i = 0; i < n_; i++)
		{
			for(int j = i+1; j < n_; j++)
			{
				if(adjMat(i,j) != 0)
				{
				}
			}
		}
	}
};

int main()
{
	using std::ios;
	using std::sqrt;
	constexpr int N = 20;
	
	const int alpha = 1;
	
	std::random_device rd;
	std::default_random_engine re{rd()};


	auto st = rs.generateRandomState(N);

	std::cout << st.norm() << std::endl;

	OverlapOptimizer oo(N, st);
	yannq::RBM<std::complex<double> > rbm(N, 3*N);

	rbm.initializeRandom(re);
	Adam<double> adam{};


	for(int ll = 0; ll <= 5000; ++ll)
	{
		if(ll%10 == 0)
		{
			char fileName[30];
			sprintf(fileName, "w%04d.dat",ll);
			std::fstream out(fileName, ios::binary|ios::out);
			{
				boost::archive::binary_oarchive oa(out);
				oa << rbm;
			}

		}

		auto f = oo.gradientExact(rbm);
		Eigen::VectorXd up = adam.getUpdate(f);

		rbm.updateParams(Eigen::Map<Eigen::VectorXcd>((std::complex<double>*)up.data(), rbm.getDim(), 1));
		std::cout << oo.logOverlap(rbm) << std::endl;

	}


	return 0;
}
