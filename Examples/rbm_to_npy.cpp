#include <filesystem>
#include <cnpy.h>

#include "Machines/RBM.hpp"
#include "Serializers/SerializeRBM.hpp"

int main(int argc, char* argv[])
{
	using Machine = yannq::RBM<std::complex<double> >;
	namespace fs = std::filesystem;
	if(argc != 2)
	{
		printf("Usage: %s [rbm data file.dat]\n", argv[0]);
		return 1;
	}
	
	auto dataFile = fs::path(argv[1]);

	if(!fs::exists(dataFile))
	{
		printf("File %s does not exists!\n", dataFile.c_str());
		return 1;
	}

	auto qs = std::unique_ptr<Machine>(nullptr);
	std::fstream in(dataFile, std::ios::binary | std::ios::in);
	{
		cereal::BinaryInputArchive oa(in);
		oa(qs);
	}
	auto npyFile = dataFile.replace_extension(".npz");


	cnpy::npz_save(npyFile.c_str(), "a", qs->getA().data(), {qs->getN()}, "w");
	cnpy::npz_save(npyFile.c_str(), "b", qs->getB().data(), {qs->getM()}, "a");
	cnpy::npz_save(npyFile.c_str(), "w", qs->getW().data(), {qs->getM(), qs->getN()}, "a");

	return 0;

}
