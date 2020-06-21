#ifndef HAMILTONIANS_KITAEVHEX_HPP
#define HAMILTONIANS_KITAEVHEX_HPP
#include <Eigen/Eigen>
#include <nlohmann/json.hpp>

class KitaevHex
{
private:
	const int n_;
	const int m_;
	const double Jx_;
	const double Jy_;
	const double Jz_;

	const double h_;

public:

	KitaevHex(int n, int m, double J, double h = 0.0)
		: n_(n), m_(m), Jx_(J), Jy_(J), Jz_(J), h_(h)
	{
		assert(n % 2 == 0);
		assert(m % 2 == 0);
	}

	KitaevHex(int n, int m, double Jx, double Jy, double Jz)
		: n_(n), m_(m), Jx_(Jx), Jy_(Jy), Jz_(Jz), h_{0.0}
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
			{"Jx", Jx_},
			{"Jy", Jy_},
			{"Jz", Jz_},
			{"h", h_},
			{"n", n_},
			{"m", m_},
		};
	}
	
	template<class State>
	typename State::Scalar operator()(const State& smp) const
	{
		typename State::Scalar s = 0.0;
		constexpr std::complex<double> I(0.,1.);

		for(int i = 0; i < n_*m_; ++i)
		{
			//s += h_*smp.sigmaAt(i); //hz
			s += h_*smp.ratio(i); //hx
		}

		for(auto &xx: xLinks())
		{
			s += Jx_*smp.ratio(xx.first, xx.second); //xx
		}
		for(auto &yy: yLinks())
		{
			int zzval = smp.sigmaAt(yy.first)*smp.sigmaAt(yy.second);
			s += -Jy_*zzval*smp.ratio(yy.first, yy.second); //yy
		}
		for(auto &zz: zLinks())
		{
			int zzval = smp.sigmaAt(zz.first)*smp.sigmaAt(zz.second);
			s += Jz_*zzval; //yy
		}
		return s;
	}

	std::map<uint32_t, double> operator()(uint32_t col) const
	{
		std::map<uint32_t, double> m;

		for(int i = 0; i < n_*m_; ++i)
		{
			double k = (1-2*int((col >> i) & 1));
			//m[col] += h_*k; //hz
			m[col ^ (1<<i)] += h_; //hx
		}

		for(auto &xx: xLinks())
		{
			int t = (1 << xx.first) | (1 << xx.second);
			m[col ^ t] += Jx_; //xx
		}
		for(auto &yy: yLinks())
		{
			int zzval = (1-2*int((col >> yy.first) & 1))*(1-2*int((col >> yy.second) & 1));
			int t = (1 << yy.first) | (1 << yy.second);
			m[col ^ t] += -Jy_*zzval; //yy
		}
		for(auto &zz: zLinks())
		{
			int zzval = (1-2*int((col >> zz.first) & 1))*(1-2*int((col >> zz.second) & 1));
			m[col] += Jz_*zzval; //zz
		}

		return m;
	}
};
#endif//HAMILTONIANS_KITAEVHEX_HPP
