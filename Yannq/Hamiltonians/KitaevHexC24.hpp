#ifndef HAMILTONIANS_KITAEVHEXC24_HPP
#define HAMILTONIANS_KITAEVHEXC24_HPP
#include <Eigen/Eigen>
#include <nlohmann/json.hpp>

class KitaevHexC24
{
private:
	static constexpr int N = 24;
	double Jx_;
	double Jy_;
	double Jz_;

	double hx_;

public:

	KitaevHexC24(double J)
		: Jx_(J), Jy_(J), Jz_(J), hx_{}
	{
	}

	KitaevHexC24(double Jx, double Jy, double Jz)
		: Jx_(Jx), Jy_(Jy), Jz_(Jz), hx_{}
	{
	}

	KitaevHexC24(double J, double hx)
		: Jx_(J), Jy_(J), Jz_(J), hx_(hx)
	{
	}

	std::vector<std::pair<int,int> > xLinks() const
	{
		std::vector<std::pair<int,int> > res;
		//first row
		res.emplace_back(0,3);
		res.emplace_back(1,4);
		//second row
		res.emplace_back(5,9);
		res.emplace_back(6,10);
		res.emplace_back(7,11);
		//thrid row
		res.emplace_back(12,16);
		res.emplace_back(13,17);
		res.emplace_back(14,18);
		//forth row
		res.emplace_back(19,22);
		res.emplace_back(20,23);

		//between
		res.emplace_back(15,2);
		res.emplace_back(21,8);
		return res;
	}

	std::vector<std::pair<int,int> > yLinks() const
	{
		std::vector<std::pair<int,int> > res;
		//first row
		res.emplace_back(0,2);
		res.emplace_back(1,3);
		//second row
		res.emplace_back(5,8);
		res.emplace_back(6,9);
		res.emplace_back(7,10);
		//thrid row
		res.emplace_back(13,16);
		res.emplace_back(14,17);
		res.emplace_back(15,18);
		//forth row
		res.emplace_back(20,22);
		res.emplace_back(21,23);
		
		//between
		res.emplace_back(4,12);
		res.emplace_back(11,19);
		return res;
	}

	std::vector<std::pair<int,int> > zLinks() const
	{
		std::vector<std::pair<int,int> > res;
		res.emplace_back(2,5);
		res.emplace_back(3,6);
		res.emplace_back(4,7);

		res.emplace_back(8,12);
		res.emplace_back(9,13);
		res.emplace_back(10,14);
		res.emplace_back(11,15);

		res.emplace_back(16,19);
		res.emplace_back(17,20);
		res.emplace_back(18,21);
		res.emplace_back(11,15);

		res.emplace_back(0,22);
		res.emplace_back(1,23);
		return res;
	}

	nlohmann::json params() const
	{
		return nlohmann::json
		{
			{"name", "KITAEVHEXC24"},
			{"Jx", Jx_},
			{"Jy", Jy_},
			{"Jz", Jz_},
			{"hx", hx_},
		};
	}
	
	template<class State>
	typename State::Scalar operator()(const State& smp) const
	{
		typename State::Scalar s = 0.0;
		constexpr std::complex<double> I(0.,1.);

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
		for(int n = 0; n < N; n++)
		{
			s += hx_*smp.ratio(n);
		}
		return s;
	}

	std::map<uint32_t, double> operator()(uint32_t col) const
	{
		std::map<uint32_t, double> m;

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
		for(int n = 0; n < N; n++)
		{
			m[col ^ (1<<n)] += hx_;
		}

		return m;
	}
};
#endif//HAMILTONIANS_KITAEVHEXC24_HPP
