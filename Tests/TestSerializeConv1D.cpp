#define CATCH_CONFIG_MAIN
#include <sstream>
#include <random>

#include <catch.hpp>

#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>

#include <Machines/layers/Conv1D.hpp>
#include <Serializers/SerializeConv1D.hpp>

TEMPLATE_TEST_CASE("test RBM serialization", "[RBM][serialization]", 
		double, float)
{
	std::random_device rd;
	std::default_random_engine re{rd()};
	
	using Machine = yannq::Conv1D<TestType>;

	for(bool useBias: {true,false})
	{
		for(int inChannels : {1,2,3,4})
		{
			for(int outChannels : {1,2,3,4})
			{
				for(int kernelSize: {1,3,5})
				{
					auto conv = std::make_unique<Machine>(inChannels, outChannels, kernelSize, 1, useBias);
					conv->randomizeParams(re, 1.0);

					{
						std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
						{
							cereal::BinaryOutputArchive oarchive( ss );
							oarchive(conv);
						}
					
						{
							cereal::BinaryInputArchive iarchive( ss );
							std::unique_ptr<Machine> deserialized{nullptr};
							iarchive(deserialized);

							REQUIRE(*conv == *deserialized);
						}
					}
				}
			}
		}
	}
}
