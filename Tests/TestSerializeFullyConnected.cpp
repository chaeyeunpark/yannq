#define CATCH_CONFIG_MAIN
#include <sstream>
#include <random>

#include <catch.hpp>

#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>

#include <Machines/layers/FullyConnected.hpp>
#include <Serializers/SerializeFullyConnected.hpp>

TEMPLATE_TEST_CASE("Test FullyConnected serialization", "[FullyConnected][serialization]", 
		double, float)
{
	std::random_device rd;
	std::default_random_engine re{rd()};
	
	using Machine = yannq::FullyConnected<TestType>;

	for(bool useBias: {true,false})
	{
		for(int inputDim : {4,5,6,7,8})
		{
			for(int outputDim : {4,5,6,7,8})
			{
				auto fc = std::make_unique<Machine>(inputDim, outputDim, useBias);
				fc->randomizeParams(re, 1.0);

				{
					std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
					{
						cereal::BinaryOutputArchive oarchive( ss );
						oarchive(fc);
					}
				
					{
						cereal::BinaryInputArchive iarchive( ss );
						std::unique_ptr<Machine> deserialized{nullptr};
						iarchive(deserialized);

						REQUIRE(*fc == *deserialized);
					}
				}
			}
		}
	}
}
