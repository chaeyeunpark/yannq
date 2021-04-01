#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <Machines/layers/layers.hpp>
#include <Machines/FeedForward.hpp>
#include <Utilities/Utility.hpp>
#include <random>

#include "Serializers/SerializeFeedForward.hpp"

using namespace yannq;
using namespace Eigen;


TEMPLATE_TEST_CASE("test backward of FeedForward", "[feed_forward][backward]", float, double, (long double))
{
	std::random_device rd;
	std::default_random_engine re{rd()};
	{
		bool useBias = false;
		const int inputSize = 20;

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

		{
			std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
			{
				cereal::BinaryOutputArchive oarchive( ss );
				oarchive(ff);
			}
		
			{
				FeedForward<TestType> deserialized;
				cereal::BinaryInputArchive iarchive( ss );
				iarchive(deserialized);

			}
		}

	}
}
