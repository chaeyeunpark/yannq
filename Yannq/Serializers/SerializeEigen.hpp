#ifndef YANNQ_SERIALIZERS_SERIALIZEEIGEN_HPP
#define YANNQ_SERIALIZERS_SERIALIZEEIGEN_HPP
#include <Eigen/Dense>
#include <cereal/archives/binary.hpp>
namespace cereal
{
	template<   class Archive, 
				class S, 
				int Rows_, 
				int Cols_, 
				int Ops_, 
				int MaxRows_, 
				int MaxCols_>
	inline void save(
		Archive & ar, 
		const Eigen::Matrix<S, Rows_, Cols_, Ops_, MaxRows_, MaxCols_> & g )
		{
			int rows = g.rows();
			int cols = g.cols();

			ar(rows, cols, cereal::binary_data(g.data(), sizeof(S)*rows * cols));
		}

	template<   class Archive, 
				class S, 
				int Rows_,
				int Cols_,
				int Ops_, 
				int MaxRows_, 
				int MaxCols_>
	inline void load(
		Archive & ar, 
		Eigen::Matrix<S, Rows_, Cols_, Ops_, MaxRows_, MaxCols_> & g )
	{
		int rows, cols;
		ar(rows);
		ar(cols);
		g.resize(rows, cols);
		ar(cereal::binary_data(g.data(), sizeof(S)*rows * cols));
	}

} // namespace cereal
#endif//YANNQ_SERIALIZERS_SERIALIZEEIGEN_HPP
