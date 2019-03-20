#ifndef CY_EIGEN_RANDOM_STATE_HPP
#define CY_EIGEN_RANDOM_STATE_HPP
#include <Eigen/Dense>
#include <random>
#include <omp.h>

template<class RandomEngine>
class RandomState
{
private:
	int nThreads_;
	std::vector<RandomEngine> re_;

public:
	RandomState()
	{
		std::random_device rd;
#pragma omp parallel
		{
#pragma omp single
			{
				nThreads_ = omp_get_num_threads();
			}
		}
		for(int i = 0; i < nThreads_; i++)
		{
			re_.emplace_back(rd());
		}
	}

	Eigen::VectorXcd generateRandomState(const int N)
	{
		Eigen::VectorXcd res(1<<N);
#pragma omp parallel
		{
			std::normal_distribution<double> nd{};
			int tid = omp_get_thread_num();
#pragma omp for schedule(static,8)
			for(int i = 0; i < (1<<N); i++)
			{
				res(i) = std::complex<double>(nd(re_[tid]), nd(re_[tid]));
			}
		}

		res.normalize();
		return res;
	}
};
#endif//CY_EIGEN_RANDOM_STATE_HPP
