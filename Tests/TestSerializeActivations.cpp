#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include <sstream>
#include <random>

#include <Machines/layers/AbstractLayer.hpp>
#include <Machines/layers/ActivationLayer.hpp>

#include <Serializers/SerializeActivationLayers.hpp>

//template void yannq::Identity<double>::serialize<cereal::BinaryInputArchive>( cereal::BinaryInputArchive & );
//template void yannq::Identity<double>::serialize<cereal::BinaryOutputArchive>( cereal::BinaryOutputArchive & );

TEMPLATE_PRODUCT_TEST_CASE("Test serialization of parameterless activation layers",
		"[Activation][serialization]", (yannq::Identity, yannq::LnCosh, yannq::Tanh, yannq::Sigmoid, yannq::ReLU, yannq::HardTanh, yannq::SoftShrink, yannq::SoftSign), (double, float))
{
	std::random_device rd;
	std::default_random_engine re{rd()};

	using Scalar = typename TestType::Scalar;

	std::cout << cereal::detail::binding_name<TestType>::name() << std::endl;

	using Parent = yannq::AbstractLayer<Scalar>;
	{
		TestType toSave;
		std::unique_ptr<Parent>
			t = std::make_unique<TestType>(toSave) ;
		std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
		{
			cereal::BinaryOutputArchive oarchive( ss );
			oarchive(t);
		}

		{
			cereal::BinaryInputArchive iarchive( ss );
			std::unique_ptr<Parent> deserialized{nullptr};
			iarchive(deserialized);
			
			TestType* tt = dynamic_cast<TestType*>(deserialized.get());
			REQUIRE(toSave == *tt);
		}
	}
}
