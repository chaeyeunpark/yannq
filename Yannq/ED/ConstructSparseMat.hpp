#ifndef EDP_CONSTRUCTSPARSEMAT_HPP
#define EDP_CONSTRUCTSPARSEMAT_HPP
#include <Eigen/Sparse>
#include <tbb/tbb.h>
namespace edp
{
	template< class ColFunc>
	auto constructSparseMat(uint64_t dim, ColFunc&& colFunc)
	{
		using T = typename std::result_of_t<ColFunc(uint32_t)>::mapped_type;
		using TripletT = Eigen::Triplet<T>;
		std::vector<TripletT> tripletList;
		tripletList.reserve(3*dim);
		for(uint64_t col = 0; col < dim; ++col)
		{
			auto m = colFunc(col);
			for(const auto& v: m)
			{
				tripletList.emplace_back(v.first, col, v.second);
			}
		}

		Eigen::SparseMatrix<T> res(dim, dim);
		res.setFromTriplets(tripletList.begin(), tripletList.end());
		return res;
	}

	/**
	 * @basis: Random access iterable container for the basis
	 */
	template<typename ColFunc, typename RandomIterable>
	auto constructSubspaceMat(ColFunc&& t, RandomIterable&& basis)
	{
		const uint32_t n = basis.size();
		using T = typename std::result_of_t<ColFunc(uint32_t)>::mapped_type;

		using TripletT = Eigen::Triplet<T>;
		tbb::concurrent_vector<TripletT> tripletList;
		tbb::parallel_for(0u, n, 
			[&](uint32_t idx)
		{
			std::map<uint32_t, T> m = t(basis[idx]);
			auto iter = basis.begin();
			for(auto& kv: m)
			{
				iter = std::lower_bound(iter, basis.end(), kv.first);
				if(iter == basis.end())
					break;
				auto j = std::distance(basis.begin(), iter);
				tripletList.emplace_back(idx, j, kv.second);
			}
		});

		Eigen::SparseMatrix<T> res(n, n);
		res.setFromTriplets(tripletList.begin(), tripletList.end());
		return res;
	}

}

#endif//EDP_CONSTRUCTSPARSEMAT_HPP
