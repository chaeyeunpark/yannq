#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include <Machines/RBM.hpp>
//#include <Serializers/SerializeRBM.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <sstream>
#include <random>

TEMPLATE_TEST_CASE("test RBM serialization", "[RBM][serialization]", 
		std::complex<double>, double)
{
	std::random_device rd;
	std::default_random_engine re{rd()};
	std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
	
	using Machine = yannq::RBM<TestType>;

	for(bool useBias: {true,false})
	{
		auto rbm = std::make_unique<Machine>(20, 40, useBias);
		rbm->initializeRandom(re, 0.1);

		{
			cereal::BinaryOutputArchive oarchive( ss );
			oarchive(rbm);
		}
		
		{
			cereal::BinaryInputArchive iarchive( ss );
			std::unique_ptr<Machine> deserialized{nullptr};
			iarchive(deserialized);

			REQUIRE(*rbm == *deserialized);
		}
	}
}
