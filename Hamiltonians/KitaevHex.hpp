#ifndef HAMILTONIANS_KITAEVHEX_HPP
#define HAMILTONIANS_KITAEVHEX_HPP
#include <Eigen/Eigen>
#include <nlohmann/json.hpp>

class KitaevHex
{
private:
	const int n_;
	const int m_;
	const double J_;

public:

	KitaevHex(int n, int m, double J)
		: n_(n), m_(m), J_(J)
	{
		assert(n % 2 == 0);
		assert(m % 2 == 0);
	}

	int blackIdx(int row, int col) const
	{
		const int k = n_/2;
		row = ((row%m_) + m_) % m_;
		col = ((col%k) + k) % k;
		return row*n_ + col;
	}
	int whiteIdx(int row, int col) const
	{
		const int k = n_/2;
		row = ((row%m_) + m_) % m_;
		col = ((col%k) + k) % k;
		return row*n_ + col + (n_/2);
	}

	std::vector<std::pair<int,int> > xLinks() const
	{
		std::vector<std::pair<int,int> > res;
		const int k = n_/2;
		for(int row = 0; row < m_; row += 2)
		{
			for(int i = 0; i < k; i++)
			{
				res.emplace_back(blackIdx(row, i), whiteIdx(row, i-1));
			}
		}
		for(int row = 1; row < m_; row += 2)
		{
			for(int i = 0; i < k; i++)
			{
				res.emplace_back(blackIdx(row,i), whiteIdx(row,i));
			}
		}
		return res;
	}

	std::vector<std::pair<int,int> > yLinks() const
	{
		std::vector<std::pair<int,int> > res;
		const int k = n_/2;
		for(int row = 0; row < m_; row += 2)
		{
			for(int i = 0; i < k; i++)
			{
				res.emplace_back(blackIdx(row, i), whiteIdx(row,i));
			}
		}
		for(int row = 1; row < m_; row += 2)
		{
			for(int i = 0; i < k; i++)
			{
				res.emplace_back(blackIdx(row, i), whiteIdx(row,i+1));
			}
		}

		return res;
	}

	std::vector<std::pair<int,int> > zLinks() const
	{
		std::vector<std::pair<int,int> > res;
		const int k = n_/2;
		for(int row = 0; row < m_; row++)
		{
			for(int i = 0; i < k; i++)
			{
				res.emplace_back(whiteIdx(row, i), blackIdx(row+1,i));
			}
		}
		return res;
	}

	nlohmann::json params() const
	{
		return nlohmann::json
		{
			{"name", "KITAEVHEX"},
			{"n", n_},
			{"m", m_},
		};
	}
	
	template<class State>
	typename State::T operator()(const State& smp) const
	{
		typename State::T s = 0.0;

		for(auto &xx: xLinks())
		{
			s += J_*smp.ratio(xx.first, xx.second); //xx
		}
		for(auto &yy: yLinks())
		{
			int zzval = smp.sigmaAt(yy.first)*smp.sigmaAt(yy.second);
			s += -J_*zzval*smp.ratio(yy.first, yy.second); //yy
		}
		for(auto &zz: zLinks())
		{
			int zzval = smp.sigmaAt(zz.first)*smp.sigmaAt(zz.second);
			s += J_*zzval; //yy
		}
		return s;
	}

	std::map<uint32_t, double> operator()(uint32_t col) const
	{
		std::map<uint32_t, double> m;
		for(auto &xx: xLinks())
		{
			int t = (1 << xx.first) | (1 << xx.second);
			m[col ^ t] += J_; //xx
		}
		for(auto &yy: yLinks())
		{
			int zzval = (1-2*(col >> yy.first))*(1-2*(col >> yy.second));
			int t = (1 << xx.first) | (1 << xx.second);
			m[col ^ t] += -J_*zzval; //yy
		}
		for(auto &zz: zLinks())
		{
			int zzval = (1-2*(col >> zz.first))*(1-2*(col >> zz.second));
			m[col] += J_*zzval; //yy
		}

		return m;
	}
};
#endif//HAMILTONIANS_KITAEVHEX_HPP
