#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <ios>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <nlohmann/json.hpp>

#include "Machines/RBM.hpp"
#include "Serializers/SerializeRBM.hpp"

#include "Hamiltonians/XXZ.hpp"

#include "Optimizers/OptimizerFactory.hpp"

#include "SRMatExactBasis.hpp"
#include "SROptimizerCG.hpp"


using namespace yannq;
using std::ios;

std::vector<uint32_t> generateBasis(int n, int nup)
{
	std::vector<uint32_t> basis;
	uint32_t v = (1u<<nup)-1;
	uint32_t w;
	while(v < (1u<<n))
	{
		basis.emplace_back(v);
		uint32_t t = v | (v-1);
		w = (t + 1) | (((~t & -~t) - 1) >> (__builtin_ctz(v) + 1));
		v = w;
	}
	return basis;
}

template<typename T>
double cosBetween(const Eigen::Matrix<T, Eigen::Dynamic, 1>& g1, const Eigen::Matrix<T, Eigen::Dynamic, 1>& g2)
{
	double t = g1.real().transpose()*g2.real();
	t += g1.imag().transpose()*g2.imag();
	return t/(g1.norm()*g2.norm());
}


template<typename Machine>
class InterpolateRBM
{
public:
	using Vector = typename Machine::Vector;
	using RealVector = Eigen::Matrix<typename remove_complex<typename Machine::ScalarType>::type, Eigen::Dynamic, 1>;
	using Matrix = typename Machine::Matrix;

	using T = typename Machine::ScalarType;
	using RealT = typename remove_complex<typename Machine::ScalarType>::type;
private:
	RealVector w0_;
	RealVector errors_;
	RealVector errors2_;
	RealVector dir1_;
	RealVector dir2_;

	double n1_;
	double n2_;


public:
	static RealVector toRealVec(const Vector& v) 
	{
		return Eigen::Map<RealVector>((RealT*)v.data(), 2*v.rows(), 1);
	}
	static Vector reverse(const RealVector& v) 
	{
		return Eigen::Map<Vector>((T*)v.data(), v.rows()/2, 1);
	}

	InterpolateRBM(const Matrix& weights)
	{
		w0_ = toRealVec(weights.col(0));
		errors_.resize(weights.cols());
		errors2_.resize(weights.cols());
		dir1_ = toRealVec(weights.col(weights.cols() - 1) - weights.col(0));
		n1_ = dir1_.norm();
		dir1_ /= n1_;

		double max = std::numeric_limits<double>::min();
		int maxIdx;
		for(int i = 0; i < weights.cols(); i++)
		{
			RealVector r = toRealVec(weights.col(i) - weights.col(0));
			r -= (dir1_.transpose()*r)*dir1_;
			errors_[i] = r.norm();
			if(max < errors_[i])
			{
				max = errors_[i];
				maxIdx = i;
			}
		}
		dir2_ = toRealVec(weights.col(maxIdx) - weights.col(0));
		dir2_ -= (dir1_.transpose()*dir2_)*dir1_;
		n2_ = dir2_.norm();
		dir2_ /= n2_;
		
		for(int i = 0; i < weights.cols(); i++)
		{
			RealVector r = toRealVec(weights.col(i) - weights.col(0));
			r -= (dir1_.transpose()*r)*dir1_;
			r -= (dir2_.transpose()*r)*dir2_;
			errors2_[i] = r.norm();
		}

	}

	RealVector getErrors() const
	{
		return errors_;
	}
	
	RealVector getErrors2() const
	{
		return errors2_;
	}

	Vector getParamAt(double alpha, double beta)
	{
		RealVector p = w0_;
		p += alpha*n1_*dir1_;
		p += beta*n2_*dir2_;
		return reverse(p);
	}


};


int main(int argc, char** argv)
{
	using namespace yannq;
	using namespace boost::filesystem;
	using std::ios;
	using nlohmann::json;

	std::random_device rd;
	std::default_random_engine re(rd());

	using ValT = std::complex<double>;

	if(argc != 2)
	{
		printf("Usage: %s [resDir]\n", argv[0]);
		return 1;
	}
	path resDir = argv[1];
	path paramPath = resDir / "params.dat";
	json paramIn;
	ifstream fin(paramPath);
	fin >> paramIn;
	fin.close();

	const int n = paramIn.at("machine").at("n").get<int>();
	const int m = paramIn.at("machine").at("m").get<int>();
	const double delta = paramIn.at("Hamiltonian").at("Delta").get<double>();

	using Machine = RBM<ValT, true>;
	Machine qs(n, m);
	Eigen::MatrixXcd weights(qs.getDim(), 601);
	for(int i = 0; i <= 600; i++)
	{
		int ll = 5*i;
		char fileName[50];
		sprintf(fileName, "w%04d.dat", ll);

		path filePath = resDir / fileName;

		fstream in(filePath, ios::binary|ios::in);
		{
			boost::archive::binary_iarchive ia(in);
			ia >> qs;
		}

		auto p = qs.getParams();
		weights.col(i) = p;
	}

	InterpolateRBM<Machine> interpolator(weights);

	/*
	auto errors = interpolator.getErrors();
	auto errors2 = interpolator.getErrors2();

	for(int i = 0; i < 600; i++)
	{
		std::cout << errors(i) << std::endl;
		std::cerr << errors2(i) << std::endl;
	}
	*/

	auto basis = generateBasis(n, n/2);
	XXZ ham(n, 1.0, delta);
	SRMatExactBasis<Machine> srex(qs, basis, ham);
	
	for(int ai = 0; ai <=  100; ai++)
	{
		for(int bi = 0; bi <=  100; bi++)
		{
			double alpha = ai*0.02;
			double beta = bi*0.03-1.5;
			auto p = interpolator.getParamAt(alpha, beta);
			qs.setParams(p);
			srex.constructExact();

			double e = srex.getEnergy();
			std::cout << alpha << "\t" << beta << "\t" << e << std::endl;
		}
	}
	return 0;
}
