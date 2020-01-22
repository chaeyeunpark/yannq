#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <Machines/layers/layers.hpp>
#include <Machines/FeedForward.hpp>
#include <Utilities/Utility.hpp>
#include <random>

using namespace yannq;
using namespace Eigen;

template<typename T>
typename yannq::AbstractLayer<T>::VectorType
numerical_grad(yannq::FeedForward<T>& ff, const VectorXi& sigma)
{
	using VectorT = typename yannq::AbstractLayer<T>::VectorType;
	using MatrixT = typename yannq::AbstractLayer<T>::MatrixType;

	const double h = 1e-5;
	
	VectorT par = ff.getParams();
	VectorT grad(par.size());
	for(int i = 0; i < par.size(); i++)
	{
		T val = par(i);
		par(i) += h;
		ff.setParams(par);
		auto res1 = ff.forward(sigma);
		
		par(i) = val - h;
		ff.setParams(par);
		auto res2 = ff.forward(sigma);

		par(i) = val;
		grad(i) = (res1.back()(0) - res2.back()(0))/(2*h);
	}
	return grad;
}

TEMPLATE_TEST_CASE("test backward of FeedForward", "[feed_forward][backward]", double, (std::complex<double>))
{
	using Catch::Matchers::Floating::WithinAbsMatcher;
	using std::abs;

	std::random_device rd;
	std::default_random_engine re{rd()};
	uint32_t inputSize = 20;

	for(bool useBias: {false, true})
	{
		FeedForward<TestType> ff;
		ff.template addLayer<Conv1D>(1, 12, 3, 1, useBias);
		ff.template addLayer<ReLU>();
		ff.template addLayer<Conv1D>(12, 10, 3, 1, useBias);
		ff.template addLayer<ReLU>();
		ff.template addLayer<Conv1D>(10, 8, 3, 1, useBias);
		ff.template addLayer<ReLU>();
		ff.template addLayer<Conv1D>(8, 6, 3, 1, useBias);
		ff.template addLayer<ReLU>();
		ff.template addLayer<Conv1D>(6, 4, 3, 1, useBias);
		ff.template addLayer<ReLU>();
		ff.template addLayer<Conv1D>(4, 2, 3, 1, useBias);
		ff.template addLayer<ReLU>();
		ff.template addLayer<FullyConnected>(inputSize*2, 1, useBias);
		ff.template addLayer<LnCosh>();

		ff.initializeRandom(re, 0.1);

		for(int i = 0; i < 10; i++)
		{
			VectorXi input = randomSigma(inputSize, re);
			auto res = ff.forward(input);
			auto grad1 = ff.backward(res);
			auto grad2 = numerical_grad(ff, input);
			/*
			if((grad1 - grad2).norm() > 1e-6)
			{
				std::cout << "input:\n" << input.transpose() << std::endl;
				std::cout << "weight: \n" << ff.getParams().transpose() << std::endl;
				std::cout << "res: \n";
				for(int i = 0; i < res.size(); ++i)
				{
					std::cout << "layer" << i << ": " << res[i].transpose() << std::endl;
				}
				auto diff = (grad1 - grad2).eval();
				for(int i = 0; i < diff.size(); i++)
				{
					if(abs(diff(i)) > 1e-4)
						std::cout << i << "\t" << grad1(i) << "\t" << grad2(i) << std::endl;
				}
			}
			*/
			REQUIRE( (grad1 - grad2).norm()/grad1.size() < 1e-5);
		}
	}
}
