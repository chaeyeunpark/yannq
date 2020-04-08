#include <Machines/RBM.hpp>
#include <Serializers/SerializeEigen.hpp>
#include <Serializers/SerializeRBM.hpp>
#include <filesystem>
struct OldRBM
{
	int n, m;
	Eigen::VectorXcd A, B;
	Eigen::MatrixXcd W;

	template<class Archive>
	void serialize(Archive & archive)
	{
		archive(n,m);
		archive(A,B,W);
	}
};

int main(int argc, char *argv[])
{
	if(argc != 3)
	{
		printf("Usage: %s [old_file] [new_file]\n", argv[0]);
		return 1;
	}

	using std::filesystem::path;

	auto old_file = path(argv[1]);
	auto new_file = path(argv[2]);

	std::unique_ptr<OldRBM> oldRBM{nullptr};
	{
		std::ifstream fin(old_file, std::ios::binary);
		cereal::BinaryInputArchive ia(fin);
		ia >> oldRBM;
	}
	using RBM = yannq::RBM<std::complex<double> >;
	auto newRBM = std::make_unique<RBM>(oldRBM->n, oldRBM->m, true);
	newRBM->setA(oldRBM->A);
	newRBM->setB(oldRBM->B);
	newRBM->setW(oldRBM->W);
	{
		std::ofstream fout(new_file, std::ios::binary);
		cereal::BinaryOutputArchive oa(fout);
		oa << newRBM;
	}
	return 0;
}
