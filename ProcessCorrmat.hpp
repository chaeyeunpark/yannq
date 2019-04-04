#ifndef CY_PROCESSCORRMAT_HPP
#define CY_PROCESSCORRMAT_HPP

#include <regex>
#include <ios>
#include <boost/filesystem.hpp>
#include <boost/range.hpp>

#include <Eigen/Eigenvalues> 

#include "SROptimizerCG.hpp"


template<class Machine, class Hamiltonian, class Sampler, class Randomizer>
class CorrMatProcessor
{
private:
	Machine& qs_;
	Hamiltonian& ham_;
	Sampler& sampler_;

	std::ofstream energyOut_;
	std::regex wRgx_;

	Randomizer randomizer_;

public:
	CorrMatProcessor(Machine& qs, Hamiltonian& ham, Sampler& sampler, Randomizer& randomizer)
		:qs_(qs), ham_(ham), sampler_(sampler), 
			wRgx_("^w([0-9]{4}).dat$", std::regex::extended), randomizer_(randomizer)
	{
		sampler_.initializeRandomEngine();

		char outName[] = "Energy.dat";
		energyOut_.open(outName, std::ios::out);
		energyOut_ << std::setprecision(10);
	}

	~CorrMatProcessor()
	{
		energyOut_.close();
	}

	bool processFile(int idx, const boost::filesystem::path& filePath)
	{
		using std::ios;
		using namespace boost::filesystem;
		if(!is_regular_file(filePath))
		{
			fprintf(stderr, "Error: cannot open %s\n", filePath.string().c_str());
			return false;
		}

		std::string fileName = filePath.filename().string();

		std::cout << "Opening " << fileName << std::endl;
		fstream qsIn(filePath, ios::binary|ios::in);
		if(qsIn.fail())
		{
			return false;
		}
		{
			boost::archive::binary_iarchive ia(qsIn);
			ia >> qs_;
		}
		qsIn.close();

		const int dim = qs_.getDim();
		std::cout << "hasNaN?: " << qs_.hasNaN() << std::endl;

		randomizer_(sampler_);
		auto sr = sampler_.sampling(2*dim, int(0.2*2*dim));

		nnqs::SRMatFree<Machine> srm(qs_);
		srm.constructFromSampling(sr, ham_);

		energyOut_ << idx << "\t" << srm.getEloc() << "\t" << srm.getElocVar() << std::endl;

		auto m = srm.corrMat();

		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es;
		es.compute(m, Eigen::EigenvaluesOnly);

		char outputName[50];
		sprintf(outputName, "EV_W%04d.dat", idx);

		fstream out(outputName, ios::out);

		out << std::setprecision(16);
		out << es.eigenvalues().transpose() << std::endl;
		out.close();
		return true;
	}

	void processAll(const boost::filesystem::path& dirPath)
	{
		using namespace boost::filesystem;
		using std::ios;


		for(auto& entry: boost::make_iterator_range(directory_iterator(dirPath), {}))
		{
			path filePath = entry.path();

			std::smatch what;
			bool matched = std::regex_match(filePath.filename().string(), what, wRgx_);
			if(!matched)
			{
				continue;
			}

			int ll;
			std::string wStr = what[1].str();
			sscanf(wStr.c_str(), "%d", &ll);

			processFile(ll, filePath);
		}
	}

	void processIdxs(const boost::filesystem::path& dirPath, const std::vector<int>& idxs)
	{
		using namespace boost::filesystem;
		for(int idx: idxs)
		{
			char fileName[50];
			sprintf(fileName, "w%04d.dat", idx);
			
			path filePath = dirPath / fileName;

			processFile(idx, filePath);
		}
	}
};
#endif//CY_PROCESSCORRMAT_HPP
