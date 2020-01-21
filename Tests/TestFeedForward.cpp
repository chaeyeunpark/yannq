#include <Machines/layers/layers.hpp>
#include <Machines/FeedForward.hpp>
#include <Utilities/Utility.hpp>
#include <random>

int main()
{
	using namespace yannq;
	using namespace Eigen;
	using dtype = double;

	std::random_device rd;
	std::default_random_engine re{rd()};
	FeedForward<dtype> ff;
	uint32_t inputSize = 20;
	ff.initializeRandom(re, 0.01);
	ff.addLayer<Conv1D>(1, 12, 3);
	ff.addLayer<LayerReLU>();
	ff.addLayer<Conv1D>(10, 8, 3);
	ff.addLayer<LayerReLU>();
	ff.addLayer<Conv1D>(8, 6, 3);
	ff.addLayer<LayerReLU>();
	ff.addLayer<Conv1D>(6, 4, 3);
	ff.addLayer<LayerReLU>();
	ff.addLayer<Conv1D>(4, 2, 3);
	ff.addLayer<LayerReLU>();
	ff.addLayer<FullyConnected>(40, 1);
	ff.addLayer<LayerLnCosh>();

	VectorXi input = randomSigma(20, re);
	auto res = ff.forward(input);


	return 0;
}
