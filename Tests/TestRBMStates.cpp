#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"

#include <algorithm>

#include "Machines/RBM.hpp"
#include "States/RBMState.hpp"
#include "States/RBMStateMT.hpp"
#include "Utilities/Utility.hpp"


using cx_double = std::complex<double>;
constexpr double eps = 1e-6;

std::vector<int> randomArray(int N, int size)
{
	std::vector<int> s;
	for(int i = 0; i < N; ++i)
	{
		s.emplace_back(i);
	}
	std::random_shuffle(s.begin(), s.end());
	s.resize(size);
	return s;
}

template<typename T, class Enable = void>
struct IsSameRatio;

template<typename T>
struct IsSameRatio<T, std::enable_if_t<yannq::is_complex_type<T>::value> > 
	: public Catch::MatcherBase<T> 
{
	using RealT = typename yannq::remove_complex<T>::type;
	T val;
	RealT tol;

    IsSameRatio(T _val, RealT _tol)
		: val(_val), tol(_tol)
	{
	}

    // Performs the test for this matcher
    bool match(const T& val2 ) const override 
	{
		T comp = val - val2;
		RealT t = comp.imag()/(2*M_PI);
		((RealT*)&comp)[1] -= std::round(t)*2*M_PI;
		return (std::abs(comp) < tol);
    }

    // Produces a string describing what this matcher does. It should
    // include any provided data (the begin/ end in this case) and
    // be written as if it were stating a fact (in the output it will be
    // preceded by the value under test).
    virtual std::string describe() const override {
        std::ostringstream ss;
        ss << "is within (" << val << ", " << tol << ") ";
        return ss.str();
    }
};

template<typename T>
struct IsSameRatio<T, std::enable_if_t<!yannq::is_complex_type<T>::value> > 
	: public Catch::MatcherBase<T>
{
	T val;
	T tol;

    IsSameRatio(T _val, T _tol)
		: val(_val), tol(_tol)
	{
	}

    // Performs the test for this matcher
    bool match(const T& val2 ) const override 
	{
		T comp = val - val2;
		return (std::abs(comp) < tol);
    }

    // Produces a string describing what this matcher does. It should
    // include any provided data (the begin/ end in this case) and
    // be written as if it were stating a fact (in the output it will be
    // preceded by the value under test).
    virtual std::string describe() const override {
        std::ostringstream ss;
        ss << "is ratio is within (" << val << ", " << tol << ") ";
        return ss.str();
    }
};


template<typename T, template<typename> class RBMState>
void TestRBMState(bool useBias)
{
	using namespace yannq;
	const int N = 10;
	using Machine = RBM<T>;
	using IsSameT = IsSameRatio<T>;
	using IsSameRealT = IsSameRatio<typename yannq::remove_complex<T>::type>;

	Machine rbm(N, N, useBias);
	std::random_device rd;
	std::default_random_engine re{0};
	std::uniform_int_distribution<int> uid{0,N-1};

	SECTION("Test single ratio")
	{
		for(int i = 0; i < 100; i++)
		{
			rbm.initializeRandom(re, 1.0);
			auto psi = getPsi(rbm, true);

			for(int k = 0; k < 10; k++)
			{
				Eigen::VectorXi sigma = randomSigma(N, re);
				uint32_t v = toValue(sigma);
				RBMState<T> st(rbm, sigma);

				for(int m = 0; m < 5; m++)
				{
					int toFlip = uid(re);
					auto logRat = std::log(psi(v ^ (1<<toFlip))) - std::log(psi(v));
					REQUIRE_THAT( st.logRatio(toFlip), IsSameT(logRat, eps));
					REQUIRE_THAT( st.logRatioRe(toFlip), IsSameRealT(std::real(logRat), eps));
				}
			}
		}
	}
	SECTION("Test double ratio")
	{
		for(int i = 0; i < 100; i++)
		{
			rbm.initializeRandom(re, 1.0);
			auto psi = getPsi(rbm, true);

			for(int k = 0; k < 30; k++)
			{
				Eigen::VectorXi sigma = randomSigma(N, re);
				uint32_t v = toValue(sigma);
				RBMState<T> st(rbm, sigma);

				for(int m = 0; m < 10; m++)
				{
					int toFlip = uid(re);
					int toFlip2 = uid(re);
					if(toFlip == toFlip2)
						continue;
					auto logRat = std::log(psi(v ^ (1<<toFlip) ^ (1<<toFlip2))) - std::log(psi(v));
					REQUIRE_THAT( st.logRatio(toFlip, toFlip2), IsSameT(logRat, eps));
					REQUIRE_THAT( st.logRatioRe(toFlip, toFlip2), IsSameRealT(std::real(logRat), eps));
				}
			}
		}
	}
	SECTION("Test vector ratio")
	{
		for(int i = 0; i < 100; i++)
		{
			rbm.initializeRandom(re, 1.0);
			auto psi = getPsi(rbm, true);

			for(int k = 0; k < 10; k++)
			{
				Eigen::VectorXi sigma = randomSigma(N, re);
				uint32_t v = toValue(sigma);
				RBMState<T> st(rbm, sigma);

				for(int m = 1; m < 6; m++)
				{
					std::vector<int> toFlip = randomArray(N, m);
					uint32_t vp = v;
					for(int e: toFlip)
					{
						vp ^= (1 << e);
					}
					auto logRat = std::log(psi(vp)) - std::log(psi(v));
					REQUIRE_THAT( st.logRatio(toFlip), IsSameT(logRat, eps));
					REQUIRE_THAT( st.logRatioRe(toFlip), IsSameRealT(std::real(logRat), eps));
				}
			}
		}
	}

	/* As State class uses cached data,
	 * we need to check flip update cache corretly
	 */
	SECTION("Test flip")
	{
		for(int i = 0; i < 100; i++)
		{
			rbm.initializeRandom(re, 1.0);
			auto psi = getPsi(rbm, true);

			for(int k = 0; k < 30; k++)
			{
				Eigen::VectorXi sigma = randomSigma(N, re);
				RBMStateValue<T> st(rbm, sigma);

				for(int m = 0; m < 100; m++)
				{
					int toFlip = uid(re);
					st.flip(toFlip);
				}
				auto theta1 = rbm.calcTheta(st.getSigma());
				auto theta2 = st.getTheta();
				REQUIRE((theta1 - theta2).norm() < eps);
			}
		}
	}
}

TEMPLATE_TEST_CASE("Test coeffs for RBM", "[RBM]", double, cx_double)
{
	TestRBMState<TestType, yannq::RBMStateValue>(true);
	TestRBMState<TestType, yannq::RBMStateValue>(false);

	TestRBMState<TestType, yannq::RBMStateValueMT>(true);
	TestRBMState<TestType, yannq::RBMStateValueMT>(false);
}

