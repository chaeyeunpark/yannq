#ifndef YANNQ_TESTS_HAMILTONIAN_HPP
#define YANNQ_TESTS_HAMILTONIAN_HPP

class HamExp
{
public:
	std::map<uint32_t, double> m_;

	HamExp()
	{
	}

	HamExp(double a)
	{
		m_[0] = a;
	}

	HamExp& operator+=(const HamExp& rhs)
	{
		for(const auto &elt: rhs.m_)
		{
			m_[elt.first] += elt.second;
		}
		return *this;
	}

	HamExp& operator*=(const double c)
	{
		for(auto& v: m_)
		{
			v.second *= c;
		}
		return *this;
	}
};

HamExp operator*(const double c, HamExp rhs)
{
	rhs *=c;
	return rhs;
}

class HamState
{
private:
	uint32_t sigma_;
public:
	using T = HamExp;

	HamState(uint32_t sigma)
		: sigma_(sigma)
	{
	}

	int sigmaAt(int n) const
	{
		return 1-2*int((sigma_ >> n) & 1);
	}

	HamExp ratio(int i) const
	{
		HamExp exp;
		exp.m_[ (1<<i) ] = 1.0;
		return exp;
	}

	HamExp ratio(int i, int j) const
	{
		HamExp exp;
		exp.m_[ (1<<i) | (1<<j) ] = 1.0;
		return exp;
	}

	std::map<uint32_t, double> eval(const HamExp& exp)
	{
		std::map<uint32_t, double> res;
		for(const auto& elt: exp.m_)
		{
			res[sigma_ ^ elt.first] = elt.second;
		}
		return res;
	}
};

template <class Hamiltonian>
std::map<uint32_t, double> getColFromStates(Hamiltonian&& ham, uint32_t col)
{
	HamState hs(col);
	HamExp exp = ham(hs);
	return hs.eval(exp);
}

double diffMap(const std::map<uint32_t, double>& lhs, const std::map<uint32_t, double>& rhs)
{
	std::set<uint32_t> keys;
	for(auto p: lhs)
	{
		keys.emplace(p.first);
	}
	for(auto p: lhs)
	{
		keys.emplace(p.first);
	}
	double diff = 0.0;
	for(auto key: keys)
	{
		diff += pow(lhs.at(key)-rhs.at(key),2);
	}
	return diff;
}

#endif//YANNQ_TESTS_HAMILTONIAN_HPP
