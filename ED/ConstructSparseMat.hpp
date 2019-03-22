#ifndef EDP_CONSTRUCTSPARSEMAT_HPP
#define EDP_CONSTRUCTSPARSEMAT_HPP
#include <Eigen/Sparse>
namespace edp
{
	template<typename T, class ColFunc>
	Eigen::SparseMatrix<T> constructSparseMat(uint64_t dim, ColFunc&& colFunc)
	{
		using TripletT = Eigen::Triplet<T>;
		std::vector<TripletT> tripletList;
		tripletList.reserve(3*dim);
		for(uint64_t col = 0; col < dim; ++col)
		{
			auto m = colFunc(col);
			for(const auto& v: m)
			{
				tripletList.push_back(TripletT(v.first, col, v.second));
			}
		}

		Eigen::SparseMatrix<T> res(dim, dim);
		res.setFromTriplets(tripletList.begin(), tripletList.end());
		return res;
	}
}

#endif//EDP_CONSTRUCTSPARSEMAT_HPP
