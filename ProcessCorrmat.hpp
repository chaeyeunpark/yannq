#ifndef CY_PROCESSCORRMAT_HPP
#define CY_PROCESSCORRMAT_HPP

#include <regex>
#include <ios>
#include <boost/filesystem.hpp>
#include <boost/range.hpp>

#include <Eigen/Eigenvalues> 

#include "SROptimizerCG.hpp"

template<class Machine, class Hamiltonian, class Sampler>
void processCorrmat(const boost::filesystem::path& dirPath, Machine& qs, Hamiltonian& ham, Sampler& sampler)
{
	using namespace boost::filesystem;
	using std::ios;

	const int dim = qs.getDim();

	char outName[] = "Energy.dat";
	std::fstream eDat(outName, ios::out);
	eDat << std::setprecision(10);
	
	std::regex reExp("^w([0-9]{4}).dat$", std::regex::extended);

	sampler.initializeRandomEngine();

	for(auto& entry: boost::make_iterator_range(directory_iterator(dirPath), {}))
	{
		if(!is_regular_file(entry))
			continue;
		
		path filePath = entry.path();
		std::string fileName = filePath.filename().string();

		std::smatch what;
		bool matched = std::regex_match(fileName, what, reExp);
		if(!matched)
			continue;

		std::string wStr = what[1].str();
		int ll;

		sscanf(wStr.c_str(), "%d", &ll);

		std::cout << "Opening " << fileName << std::endl;
		fstream in(filePath, ios::binary|ios::in);
		{
			boost::archive::binary_iarchive ia(in);
			ia >> qs;
		}
		std::cout << "hasNaN?: " << qs.hasNaN() << std::endl;

		sampler.randomizeSigma();
		auto sr = sampler.sampling(2*dim, int(0.2*2*dim));

		nnqs::SRMatFree<Machine> srm(qs);
		srm.constructFromSampling(sr, ham);

		eDat << ll << "\t" << srm.getEloc() << "\t" << srm.getElocVar() << std::endl;

		auto m = srm.corrMat();

		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es;
		es.compute(m, Eigen::EigenvaluesOnly);

		char outputName[50];
		sprintf(outputName, "EV_W%04d.dat", ll);

		std::fstream out(outputName, ios::out);

		out << std::setprecision(16);
		out << es.eigenvalues().transpose() << std::endl;
		out.close();
	}
	eDat.close();
}

#endif//CY_PROCESSCORRMAT_HPP
