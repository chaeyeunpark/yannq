#include <Machines/AmplitudePhase.cpp>
#include <Machines/layers/layers.hpp>
#include <Machines/FeedForward.hpp>

#include <iostream>
#include <GroundState/SRMatExact.hpp>
//#include <Hamiltonians/TFIsing.hpp>
#include <Hamiltonians/XXXJ1J2.hpp>
#include <Basis/BasisJz.hpp>
//#include <Basis/BasisFull.hpp>

#include <Optimizers/SGD.hpp>

int main()
{
	using namespace yannq;
	using namespace Eigen;
	const int N = 12;
	const int M = 2*N;

	std::random_device rd;
	std::default_random_engine re{rd()};

	FeedForward<double> ff;
	const int kernel_size = 5;
	ff.template addLayer<Conv1D>(1, 12, kernel_size, 1, false);
	ff.template addLayer<LeakyReLU>();
	ff.template addLayer<Conv1D>(12, 10, kernel_size, 1, false);
	ff.template addLayer<LeakyReLU>();
	ff.template addLayer<Conv1D>(10, 8, kernel_size, 1, false);
	ff.template addLayer<LeakyReLU>();
	ff.template addLayer<Conv1D>(8, 6, kernel_size, 1, false);
	ff.template addLayer<LeakyReLU>();
	ff.template addLayer<Conv1D>(6, 4, kernel_size, 1, false);
	ff.template addLayer<LeakyReLU>();
	ff.template addLayer<Conv1D>(4, 2, kernel_size, 1, false);
	ff.template addLayer<LeakyReLU>();
	ff.template addLayer<FullyConnected>(2*N, 1, false);
	ff.template addLayer<Tanh>();
	ff.initializeRandom(re, InitializationMode::He);

	AmplitudePhase qs(N,M,std::move(ff));
	qs.initializeAmplitudeRandom(re, 0.01);

    auto opt = SGD<double>(0.012, 0.0);

	XXXJ1J2<-1> ham(N, 1.0, 0.44);
	constexpr double lambda = 0.001;
	const int dim = qs.getDim();

	SRMatExact<AmplitudePhase, std::complex<double> > srex(qs, BasisJz(N,N/2), ham);

	for(int ll = 0; ll < 1000; ll++)
	{
		srex.constructExact();

		double energy = srex.getEnergy();
		Eigen::MatrixXd corrMat = srex.corrMat().real();

		corrMat += lambda*MatrixXd::Identity(dim, dim);

		Eigen::LLT<Eigen::MatrixXd> llt(corrMat);

		VectorXd grad = srex.getF().real();
		//std::cout << grad.transpose() << std::endl;
		VectorXd s = llt.solve(grad);

		VectorXd optV = opt.getUpdate(s);

		qs.updateParams(optV);

		std::cout << ll << "\t" << energy << std::endl;
	}

	return 0;
}
