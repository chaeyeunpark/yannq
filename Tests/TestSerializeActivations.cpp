#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <sstream>
#include <random>


#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>

#include <cereal/types/polymorphic.hpp>

#include <Machines/layers/AbstractLayer.hpp>
#include <Machines/layers/ActivationLayer.hpp>

//#include <Serializers/SerializeActivationLayers.hpp>

CEREAL_REGISTER_TYPE(yannq::ActivationLayer<double, yannq::activation::Identity<double>>);

CEREAL_REGISTER_POLYMORPHIC_RELATION((yannq::AbstractLayer<double>), yannq::ActivationLayer<double, yannq::activation::Identity<double>>);

//CEREAL_REGISTER_DYNAMIC_INIT(YANNQ)
/* TEMPLATE_PRODUCT_TEST_CASE("Test serialization of parameterless activation layers",
		"[Activation][serialization]", (yannq::Identity, yannq::LnCosh, yannq::Tanh, yannq::Sigmoid, yannq::ReLU, yannq::HardTanh, yannq::SoftShrink, yannq::SoftSign), (double, float))
		*/

TEMPLATE_PRODUCT_TEST_CASE("Test serialization of parameterless activation layers",
		"[Activation][serialization]", (yannq::Identity), (double))
{
	std::random_device rd;
	std::default_random_engine re{rd()};

	using Parent = yannq::AbstractLayer<typename TestType::ScalarType>;
	{
		std::unique_ptr<Parent>
			t = std::make_unique<TestType>() ;
		std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
		{
			cereal::BinaryOutputArchive oarchive( ss );
			oarchive(t);
		}

		{
			cereal::BinaryInputArchive iarchive( ss );
			std::unique_ptr<Parent> deserialized{nullptr};
			iarchive(deserialized);

			REQUIRE(*t == *deserialized);
		}
	}
}
