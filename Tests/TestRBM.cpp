#define CATCH_CONFIG_MAIN
#include <catch.hpp>

#include <random>
#include <Machines/RBM.hpp>
#include <Utilities/Utility.hpp>

template<typename T>
struct Inferior
{
};
template<>
struct Inferior<double>
{
	using Type = float;
};
template<>
struct Inferior<std::complex<double>>
{
	using Type = std::complex<float>;
};
struct ComplexWithinRelMatcher : Catch::MatcherBase<std::complex<double>> 
{
	ComplexWithinRelMatcher(std::complex<double> target, double epsilon)
		: m_target{target}, m_epsilon{epsilon}
	{
		CATCH_ENFORCE(m_epsilon >= 0., "Relative comparison with epsilon <  0 does not make sense.");
		CATCH_ENFORCE(m_epsilon  < 1., "Relative comparison with epsilon >= 1 does not make sense.");
	}
	bool match(std::complex<double> const& matchee) const override
	{
		const auto relMargin = m_epsilon * (std::max)(std::abs(matchee), std::abs(m_target));
		return std::abs(m_target - matchee) < relMargin;
	}
	std::string describe() const override
	{
		Catch::ReusableStringStream sstr;
		sstr << "and " << m_target << " are within " << m_epsilon * 100. << "% of each other";
		return sstr.str();
	}

private:
	std::complex<double> m_target;
	double m_epsilon;
};

template<typename T>
struct MyWithinRelMatcher
{
};

template<>
struct MyWithinRelMatcher<double> : public Catch::Matchers::Floating::WithinRelMatcher
{
	MyWithinRelMatcher(double target, double eps)
		: WithinRelMatcher(target, eps)
	{
	}
};
template<>
struct MyWithinRelMatcher<std::complex<double> > 
	: public ComplexWithinRelMatcher
{
	MyWithinRelMatcher(std::complex<double> target, double eps)
		: ComplexWithinRelMatcher(target, eps)
	{
	}
};


TEMPLATE_TEST_CASE("test RBM with bias", "[rbm]", double, std::complex<double>)
{
	using namespace yannq;
	using namespace Eigen;
	using Catch::Matchers::Floating::WithinRelMatcher;
	using Catch::Matchers::Floating::WithinAbsMatcher;
	std::random_device rd;
	std::default_random_engine re{rd()};

	uint32_t n = 20;
	uint32_t m = 40;

	using Machine = RBM<TestType>;
	using Vector = typename Machine::Vector;
	using Matrix = typename Machine::Matrix;
	using MachineInferior = RBM<typename Inferior<TestType>::Type>;

	SECTION("test random RBM")
	{
		MachineInferior rbm_float(n, m, true);

		rbm_float.initializeRandom(re);
		Machine rbm_double = rbm_float;

		REQUIRE(rbm_float.template cast<TestType>() == rbm_double);

		for(uint32_t _k = 0; _k < 100; ++_k)
		{
			auto sigma = randomSigma(n, re);
			auto t1 = rbm_float.makeData(sigma);
			auto t2 = rbm_double.makeData(sigma);
			auto log_coeff1 = rbm_float.logCoeff(t1);
			auto log_coeff2 = rbm_double.logCoeff(t2);

			auto log_der1 = rbm_float.logDeriv(t1);
			auto log_der2 = rbm_double.logDeriv(t2);

			REQUIRE(std::abs(static_cast<TestType>(log_coeff1) - log_coeff2) <  1e-6);

			REQUIRE_THAT(std::exp(log_coeff1), MyWithinRelMatcher<TestType>(static_cast<TestType>(rbm_float.coeff(t1)), 1e-4));
			REQUIRE_THAT(std::exp(log_coeff2), MyWithinRelMatcher<TestType>(static_cast<TestType>(rbm_double.coeff(t2)), 1e-4));

			REQUIRE((log_der1.template cast<TestType>() - log_der2).norm() < 1e-6);
		}
	}

	SECTION("test set")
	{
		Machine rbm(n, m, true);

		for(uint32_t _k = 0; _k < 10; ++_k)
		{
			rbm.initializeRandom(re);

			Matrix w = Matrix::Random(m,n);
			Vector a = Vector::Random(n);
			Vector b = Vector::Random(m);

			rbm.setA(a);
			rbm.setB(b);
			rbm.setW(w);

			assert(rbm.getA() == a);
			assert(rbm.getB() == b);
			assert(rbm.getW() == w);
		}

	}

	SECTION("test update")
	{
		Machine rbm(n, m, true);

		for(uint32_t _k = 0; _k < 10; ++_k)
		{
			rbm.initializeRandom(re);

			Matrix w = rbm.getW();
			Vector a = rbm.getA();
			Vector b = rbm.getB();

			Matrix toUpdateW = Matrix::Random(m,n);
			Vector toUpdateA = Vector::Random(n);
			Vector toUpdateB = Vector::Random(m);

			rbm.updateW(toUpdateW);
			rbm.updateA(toUpdateA);
			rbm.updateB(toUpdateB);

			REQUIRE(w + toUpdateW == rbm.getW());
			REQUIRE(a + toUpdateA == rbm.getA());
			REQUIRE(b + toUpdateB == rbm.getB());
		}
	}

	SECTION("test conservativeResize")
	{
		for(uint32_t _k = 0; _k < 10; ++_k)
		{
			Machine rbm(n, m, true);
			rbm.initializeRandom(re);

			auto sigma = randomSigma(n, re);

			TestType coeff1 = rbm.coeff(rbm.makeData(sigma));

			rbm.conservativeResize(m + 10);
			TestType coeff2 = rbm.coeff(rbm.makeData(sigma));

			REQUIRE( std::abs(coeff1 - coeff2) < 1e-6);
		}
	}

	SECTION("test log derivative")
	{
		Machine rbm(n, m, true);

		for(uint32_t _k = 0; _k < 10; ++_k)
		{
			rbm.initializeRandom(re);
			auto sigma = randomSigma(n, re);
			auto data = rbm.makeData(sigma);
			auto logDeriv = rbm.logDeriv(data);
			const double h = 1e-6;

			{//Test W grad
				Matrix wDeriv(m, n);
				for(uint32_t i = 0; i < n; ++i)
				{
					for(uint32_t j = 0; j < m; ++j)
					{
						TestType wPrev = rbm.W(j, i);
						rbm.W(j,i) = wPrev - h;
						TestType logCoeff1 = rbm.logCoeff(rbm.makeData(sigma));
						rbm.W(j,i) = wPrev + h;
						TestType logCoeff2 = rbm.logCoeff(rbm.makeData(sigma));
						rbm.W(j,i) = wPrev;

						wDeriv(j,i) = (logCoeff2 - logCoeff1)/(2*h);
					}
				}
				REQUIRE((wDeriv - Map<Matrix>(logDeriv.data(), m, n)).norm() / (n*m) < 1e-6);
			}
			{//Test A grad
				Vector aDeriv(n);
				for(uint32_t i = 0; i < n; ++i)
				{
					TestType aPrev = rbm.A(i);
					rbm.A(i) = aPrev - h;
					TestType logCoeff1 = rbm.logCoeff(rbm.makeData(sigma));
					rbm.A(i) = aPrev + h;
					TestType logCoeff2 = rbm.logCoeff(rbm.makeData(sigma));
					rbm.A(i) = aPrev;

					aDeriv(i) = (logCoeff2 - logCoeff1)/(2*h);
				}
				REQUIRE((aDeriv - Map<Vector>(logDeriv.data() + m*n, n)).norm() / n < 1e-6);
			}

		}
	}
}

TEMPLATE_TEST_CASE("test RBM without bias", "[rbm]", double, std::complex<double>)
{
	using namespace yannq;
	using namespace Eigen;
	using Catch::Matchers::Floating::WithinRelMatcher;
	using Catch::Matchers::Floating::WithinAbsMatcher;
	std::random_device rd;
	std::default_random_engine re{rd()};

	uint32_t n = 20;
	uint32_t m = 40;

	using Machine = RBM<TestType>;
	using Vector = typename Machine::Vector;
	using Matrix = typename Machine::Matrix;
	using MachineInferior = RBM<typename Inferior<TestType>::Type>;

	SECTION("test random RBM")
	{
		MachineInferior rbm_float(n, m, false);

		rbm_float.initializeRandom(re);
		Machine rbm_double = rbm_float;

		REQUIRE(rbm_float.template cast<TestType>() == rbm_double);

		for(uint32_t _k = 0; _k < 100; ++_k)
		{
			auto sigma = randomSigma(n, re);
			auto t1 = rbm_float.makeData(sigma);
			auto t2 = rbm_double.makeData(sigma);
			auto log_coeff1 = rbm_float.logCoeff(t1);
			auto log_coeff2 = rbm_double.logCoeff(t2);

			auto log_der1 = rbm_float.logDeriv(t1);
			auto log_der2 = rbm_double.logDeriv(t2);

			REQUIRE(std::abs(static_cast<TestType>(log_coeff1) - log_coeff2) <  1e-6);

			REQUIRE_THAT(std::exp(log_coeff1), MyWithinRelMatcher<TestType>(static_cast<TestType>(rbm_float.coeff(t1)), 1e-4));
			REQUIRE_THAT(std::exp(log_coeff2), MyWithinRelMatcher<TestType>(static_cast<TestType>(rbm_double.coeff(t2)), 1e-4));

			REQUIRE((log_der1.template cast<TestType>() - log_der2).norm() < 1e-6);
		}
	}

	SECTION("test set")
	{
		Machine rbm(n, m, false);

		for(uint32_t _k = 0; _k < 10; ++_k)
		{
			rbm.initializeRandom(re);

			Matrix w = Matrix::Random(m,n);

			rbm.setW(w);

			assert(rbm.getW() == w);
		}

	}

	SECTION("test update")
	{
		Machine rbm(n, m, false);

		for(uint32_t _k = 0; _k < 10; ++_k)
		{
			rbm.initializeRandom(re);

			Matrix w = rbm.getW();
			Matrix toUpdateW = Matrix::Random(m,n);
			rbm.updateW(toUpdateW);

			REQUIRE(w + toUpdateW == rbm.getW());
		}
	}

	SECTION("test conservativeResize")
	{
		for(uint32_t _k = 0; _k < 10; ++_k)
		{
			Machine rbm(n, m, false);
			rbm.initializeRandom(re);

			auto sigma = randomSigma(n, re);

			TestType coeff1 = rbm.coeff(rbm.makeData(sigma));

			rbm.conservativeResize(m + 10);
			TestType coeff2 = rbm.coeff(rbm.makeData(sigma));

			REQUIRE( std::abs(coeff1 - coeff2) < 1e-6);
		}
	}

	SECTION("test log derivative")
	{
		Machine rbm(n, m, false);

		for(uint32_t _k = 0; _k < 10; ++_k)
		{
			rbm.initializeRandom(re);
			auto sigma = randomSigma(n, re);
			auto data = rbm.makeData(sigma);
			auto logDeriv = rbm.logDeriv(data);
			const double h = 1e-6;

			{//Test W grad
				Matrix wDeriv(m, n);
				for(uint32_t i = 0; i < n; ++i)
				{
					for(uint32_t j = 0; j < m; ++j)
					{
						TestType wPrev = rbm.W(j, i);
						rbm.W(j,i) = wPrev - h;
						TestType logCoeff1 = rbm.logCoeff(rbm.makeData(sigma));
						rbm.W(j,i) = wPrev + h;
						TestType logCoeff2 = rbm.logCoeff(rbm.makeData(sigma));
						rbm.W(j,i) = wPrev;

						wDeriv(j,i) = (logCoeff2 - logCoeff1)/(2*h);
					}
				}
				REQUIRE((wDeriv - Map<Matrix>(logDeriv.data(), m, n)).norm() / (n*m) < 1e-6);
			}

		}
	}
}
