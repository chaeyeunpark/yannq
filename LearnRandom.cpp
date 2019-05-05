#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <boost/archive/binary_oarchive.hpp>
#include <ios>

#include "Machines/RBM.hpp"
#include "Optimizers/Adam.hpp"
#include "Serializers/SerializeRBM.hpp"

#include "OverlapOptimizer.hpp"
#include "RandomState.hpp"

int main()
{
	using std::ios;
	using std::sqrt;
	constexpr int N = 18;
	
	std::random_device rd;
	std::default_random_engine re{rd()};

	RandomState<std::default_random_engine> rs;

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
