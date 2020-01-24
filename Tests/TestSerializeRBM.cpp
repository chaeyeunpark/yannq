#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include <Machines/RBM.hpp>
//#include <Serializers/SerializeRBM.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <sstream>
#include <random>

TEMPLATE_TEST_CASE_SIG("test RBM serialization", "[RBM][serialization]", 
		((typename T, bool useBias), T, useBias), 
		((std::complex<double>), true), ((std::complex<double>), false), (double, true), (double, false))
{
	using TestType = yannq::RBM<T, useBias>;
	std::random_device rd;
	std::default_random_engine re{rd()};
	std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);

	auto rbm_real = std::make_unique<TestType>(20,40);
	rbm_real->initializeRandom(re, 0.1);

	{
		cereal::BinaryOutputArchive oarchive( ss );
		oarchive(rbm_real);
	}
	
	{
		cereal::BinaryInputArchive iarchive( ss );
		std::unique_ptr<TestType> deserialized{nullptr};
		iarchive(deserialized);

		REQUIRE(*rbm_real == *deserialized);
	}
}
