#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include <Machines/RBM.hpp>
#include <Serializers/SerializeRBM.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <sstream>
#include <random>

std::vector<uint32_t> diffPosition(const Eigen::VectorXcd& v1, const Eigen::VectorXcd& v2)
{
	std::vector<uint32_t> pos;
	for(uint32_t i = 0; i < v1.size(); ++i)
	{
		if(v1(i) != v2(i))
		{
			pos.push_back(i);
		}
	}
	return pos;
}

std::string hex_rep(double a)
{
	//sizeof(double) = 8 bytes 
	//unsigned int = 4 bytes
	//1 bytes in hex rep -> 2 characters
	char res[sizeof(double)*2+1];
	unsigned int* p = (unsigned int*)&a;
	sprintf(res, "%08x%08x", p[0], p[1]);
	return res;
}

std::string hex_rep(std::complex<double> a)
{
	//sizeof(double) = 16 bytes 
	//unsigned int = 4 bytes
	char res[sizeof(std::complex<double>)*2+1];
	unsigned int* p = (unsigned int*)&a;
	sprintf(res, "%08x%08x%08x%08x", p[0], p[1], p[2], p[3]);
	return res;
}

TEMPLATE_TEST_CASE("test RBM serialization", "[RBM][serialization]", 
		std::complex<double>, double)
{
	std::random_device rd;
	std::default_random_engine re{rd()};
	
	using Machine = yannq::RBM<TestType>;

	for(bool useBias: {true,false})
	{
		for(int alpha : {1,2,3,4})
		{
			auto rbm = std::make_unique<Machine>(20, alpha*20, useBias);
			rbm->initializeRandom(re, 1.0);

			{
			std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
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
			{
			std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
			{
				cereal::BinaryOutputArchive oarchive( ss );
				oarchive(*rbm);
			}
			
			{
				cereal::BinaryInputArchive iarchive( ss );
				Machine deserialized;
				iarchive(deserialized);

				REQUIRE(*rbm == deserialized);
			}
			}


		}
	}
}
