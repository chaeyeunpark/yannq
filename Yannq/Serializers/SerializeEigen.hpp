#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <cereal/archives/binary.hpp>
namespace cereal
{
	template<class Archive, class S, int Rows_, int Cols_, int Ops_, 
		int MaxRows_, int MaxCols_>
	inline void save(Archive& ar, 
		const Eigen::Matrix<S, Rows_, Cols_, Ops_, MaxRows_, MaxCols_>& g )
	{
		int rows = g.rows();
		int cols = g.cols();
		ar(rows, cols, cereal::binary_data(g.data(), sizeof(S)*rows * cols));
	}

	template<class Archive, class S, int Rows_,	int Cols_, int Ops_,
		int MaxRows_, int MaxCols_>
	inline void load(Archive & ar, 
		Eigen::Matrix<S, Rows_, Cols_, Ops_, MaxRows_, MaxCols_>& g )
	{
		int rows, cols;
		ar(rows);
		ar(cols);
		
		if((Rows_ != Eigen::Dynamic) && (Rows_ != rows)) 
			throw cereal::Exception("Saved rows mismatches with the rows of the current matrix.");

		if((Cols_ != Eigen::Dynamic) && (Cols_ != cols))
			throw cereal::Exception("Saved rows mismatches with the rows of the current matrix.");

		g.resize(rows, cols);
		ar(cereal::binary_data(g.data(), sizeof(S)*rows * cols));
	}

	template<class Archive, class S>
	void save(Archive& ar, const Eigen::SparseMatrix<S, Eigen::ColMajor>& m)
	{
		Eigen::SparseMatrix<S, Eigen::ColMajor> m_s = m;
		m_s.makeCompressed();

		int rows = m_s.rows();
		int cols = m_s.cols();
		int nnz = m_s.nonZeros();

		ar(rows, cols, nnz);
		using StorageIndex = typename Eigen::SparseMatrix<S, Eigen::ColMajor>::StorageIndex;
		
		ar(cereal::binary_data(m_s.valuePtr(), sizeof(S)*nnz));
		ar(cereal::binary_data(m_s.innerIndexPtr(), sizeof(StorageIndex)*nnz));
		ar(cereal::binary_data(m_s.outerIndexPtr(), sizeof(StorageIndex)*(cols+1)));
	}
	template<class Archive, class S>
	void load(Archive& ar, Eigen::SparseMatrix<S, Eigen::ColMajor>& m)
	{
		int rows, cols, nnz;
		ar(rows, cols, nnz);
		using StorageIndex = typename Eigen::SparseMatrix<S, Eigen::ColMajor>::StorageIndex;

		m.resize(rows, cols);

		std::vector<S> values;
		values.resize(nnz);
		std::vector<StorageIndex> innerIndices;
		innerIndices.resize(nnz);
		std::vector<StorageIndex> outerIndices;
		outerIndices.resize(cols+1);
		
		ar(cereal::binary_data(values.data(), sizeof(S)*nnz));
		ar(cereal::binary_data(innerIndices.data(), sizeof(StorageIndex)*nnz));
		ar(cereal::binary_data(outerIndices.data(), sizeof(StorageIndex)*(cols+1)));
		
		m = Eigen::Map<Eigen::SparseMatrix<S, Eigen::ColMajor> >(rows, cols, nnz,
				outerIndices.data(), innerIndices.data(), values.data());
	}
} // namespace cereal
