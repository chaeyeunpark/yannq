#ifndef CY_MPI_NODE_MV_HPP
#define CY_MPI_NODE_MV_HPP

#include <mkl.h>
#include <vector>
#include <utility>
#include <algorithm>

class NodeMV
{
private:

	std::size_t dim_;
	std::size_t rows_;

	std::vector<double> values;
	std::vector<int> colIdx;
	std::vector<int> ptrB;

	sparse_matrix_t A_;
	matrix_descr descA_;
public:

	template<class ColFunc>
	explicit NodeMV(const std::size_t dim, std::size_t row_start, std::size_t row_end, ColFunc&& col)
		: dim_(dim)
	{
		rows_ = row_end - row_start;

		auto get_first = [](const std::pair<const std::size_t, double>& p){ return p.first; };
		auto get_second = [](const std::pair<const std::size_t, double>& p){ return p.second; };
		
		ptrB.resize(rows_+1);
		ptrB[0] = 0;
		for(std::size_t i = 0; i < rows_; i++)
		{
			auto rr = col.getCol(i + row_start);
			ptrB[i+1] = ptrB[i] + rr.size();
			std::transform(rr.begin(), rr.end(), back_inserter(colIdx), get_first);
			std::transform(rr.begin(), rr.end(), back_inserter(values), get_second);
		}
		sparse_status_t m = mkl_sparse_d_create_csr(&A_, SPARSE_INDEX_BASE_ZERO, rows_, dim, 
				ptrB.data(), ptrB.data()+1, colIdx.data(), values.data());
		descA_.type = SPARSE_MATRIX_TYPE_GENERAL;

		mkl_sparse_set_mv_hint(A_, SPARSE_OPERATION_NON_TRANSPOSE, descA_, 100000);
	}

	std::size_t rows() const
	{
		return rows_;
	}

	std::size_t cols() const
	{
		return dim_;
	}

	void perform_op(double* x_in, double* y_out) const
	{
		mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A_, descA_, x_in, 0.0, y_out);
	}

};



#endif//CY_MPI_NODE_MV_HPP
