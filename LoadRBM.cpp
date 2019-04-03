#include <iostream>
#include <iomanip>
#include <fstream>
#include <ios>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <regex>

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>


#include "Machines/RBM.hpp"
#include "Serializers/SerializeRBM.hpp"

#include <nlohmann/json.hpp>

#include <cnpy.h>


template<bool useBias>
void process(
		const boost::filesystem::path& dataPath, 
		const boost::filesystem::path& resDir, 
		int n, 
		int m)
{
	using std::ios;
	using namespace boost::filesystem;
	using namespace boost;

	std::regex exp("^w([0-9]{4}).dat$", std::regex::extended);
	nnqs::RBM<std::complex<double>, useBias> qs(n, m);

	for(auto& entry: make_iterator_range(directory_iterator(dataPath), {}))
	{
		if(!is_regular_file(entry))
			continue;
		path filePath = entry.path();
		
		std::cmatch what;
		std::string fileName = filePath.filename().string();
		bool matched;
		matched = std::regex_match(fileName.c_str(), what, exp);
		if(!matched)
			continue;

		int w;
		std::string m = what[1].str();
		sscanf(m.c_str(), "%d", &w);

		fstream in(filePath, ios::binary|ios::in);
		{
			boost::archive::binary_iarchive ia(in);
			ia >> qs;
		}
		
		{
			char fileName[255];

			sprintf(fileName, "w%04d.npz", w);
			path p = resDir;
			p /= fileName;

			Eigen::MatrixXcd W = qs.getW();
			cnpy::npz_save(p.c_str(), "W", W.data(), {W.rows(), W.cols()}, "w");

			Eigen::VectorXcd A = qs.getA();
			cnpy::npz_save(p.c_str(), "A", A.data(), {A.rows()}, "a");

			Eigen::VectorXcd B = qs.getB();
			cnpy::npz_save(p.c_str(), "B", B.data(), {B.rows()}, "a");

		}
	}
}

int main(int argc, char** argv)
{
	using namespace boost::filesystem;
	using nlohmann::json;

	if(argc != 3)
	{
		printf("Usage: %s [dataDir] [resDir]\n", argv[0]);
		return 1;
	}

	path dataDir = argv[1];

	if(!is_directory(dataDir))
	{
		fprintf(stderr, "[dataDir] must be directory!\n");
		return 1;
	}

	path resDir = argv[2];

	if(!exists(resDir))
	{
		create_directory(resDir);
	}
	else
	{
		if(!is_directory(resDir))
		{
			fprintf(stderr, "[resDir] must be directory!!\n");
			return 1;
		}
	}

	
	path paramPath = dataDir;
	paramPath /= "params.dat";
	ifstream fin(paramPath);
	
	json params;
	fin >> params;

	fin.close();

	paramPath = resDir;
	paramPath /= "params.dat";
	ofstream fout(paramPath);
	fout << params;
	fout.close();

	int n = params["Machine"]["n"];
	int m = params["Machine"]["m"];

	bool useBias = params["Machine"]["useBias"];

	if(useBias)
		process<true>(dataDir, resDir,n,m);
	/*
	else
		process<false>(dataDir, resDir,n,m);
		*/

	return 0;
}
