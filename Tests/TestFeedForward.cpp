#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <Machines/layers/layers.hpp>
#include <Machines/FeedForward.hpp>
#include <Utilities/Utility.hpp>
#include <random>

#include "LayerHelper.hpp"

using namespace yannq;
using namespace Eigen;

template<typename T>
typename yannq::AbstractLayer<T>::Vector
numerical_grad(yannq::FeedForward<T>& ff, const VectorXi& sigma)
{
	using Vector = typename yannq::AbstractLayer<T>::Vector;
	using Matrix = typename yannq::AbstractLayer<T>::Matrix;

	const double h = 1e-5;
	
	Vector par = ff.getParams();
	Vector grad(par.size());
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
		grad(i) = (res1 - res2)/(2*h);
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

	bool useBias = false;
//	for(bool useBias: {false, true})
	{
		FeedForward<TestType> ff;
		ff.template addLayer<Conv1D>(1, 12, 3, 1, useBias);
		ff.template addLayer<Tanh>();
		ff.template addLayer<Conv1D>(12, 10, 3, 1, useBias);
		ff.template addLayer<Tanh>();
		ff.template addLayer<Conv1D>(10, 8, 3, 1, useBias);
		ff.template addLayer<Tanh>();
		ff.template addLayer<Conv1D>(8, 6, 3, 1, useBias);
		ff.template addLayer<Tanh>();
		ff.template addLayer<Conv1D>(6, 4, 3, 1, useBias);
		ff.template addLayer<Tanh>();
		ff.template addLayer<Conv1D>(4, 2, 3, 1, useBias);
		ff.template addLayer<Tanh>();
		ff.template addLayer<FullyConnected>(inputSize*2, 1, useBias);
		ff.initializeRandom(re, 0.1);

		for(int i = 0; i < 10; i++)
		{
			VectorXi input = randomSigma(inputSize, re);
			auto data = ff.makeData(input);
			auto grad1 = ff.backward(data);
			auto grad2 = numerical_grad(ff, input);
			REQUIRE( (grad1 - grad2).norm()/grad1.norm() < 1e-4);
		}
	}
}
