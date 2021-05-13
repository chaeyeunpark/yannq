#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <sstream>

#include <Serializers/SerializeEigen.hpp>
#include <random>

TEST_CASE("test serialize SparseMatrix", "[serialize]")
{
	const uint32_t n = 10;
	const uint32_t m = 20;

	Eigen::SparseMatrix<double> mat(n, m);
	std::uniform_int_distribution<> row_dist(0, n-1);
	std::uniform_int_distribution<> col_dist(0, n-1);
	std::normal_distribution<> val_dist;

	std::random_device rd;
	std::default_random_engine re{rd()};

	for(uint32_t k = 0; k < 10; k++)
	{
		mat.coeffRef(row_dist(re), col_dist(re)) = val_dist(re);
	}


	{
		std::stringstream ss(std::ios::binary | std::ios::in | std::ios::out);
		{
			cereal::BinaryOutputArchive oarchive( ss );
			oarchive(mat);
		}
		
		{
			cereal::BinaryInputArchive iarchive( ss );
			Eigen::SparseMatrix<double> deserialized;
			iarchive(deserialized);

			REQUIRE((mat - deserialized).norm() < 1e-6);
		}
	}
}
