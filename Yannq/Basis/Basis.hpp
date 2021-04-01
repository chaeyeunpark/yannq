#pragma once
//! \defgroup Basis Basis for a spin-1/2 system
#include "BasisJz.hpp"
#include "BasisFull.hpp"

#include <iterator>
#include <type_traits>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_sort.h>
/**
 * Enable if Iterable is not random access iterable
 */
template<class BasisType>
tbb::concurrent_vector<uint32_t> parallelConstructBasis(BasisType&& basis, std::forward_iterator_tag)
{
	tbb::concurrent_vector<uint32_t> res;
	tbb::parallel_for_each(basis.begin(), basis.end(),
		[&](uint32_t elt)
	{
		res.emplace_back(elt);
	});
	tbb::parallel_sort(res.begin(), res.end());
	return res;
}

template<class BasisType>
tbb::concurrent_vector<uint32_t> parallelConstructBasis(BasisType&& basis, std::random_access_iterator_tag)
{
	tbb::concurrent_vector<uint32_t> res(basis.size(), 0u);
	tbb::parallel_for(std::size_t(0u), basis.size(),
		[&](std::size_t idx)
	{
		res[idx] = basis[idx];
	});
	return res;
}

template<class BasisType>
inline tbb::concurrent_vector<uint32_t> parallelConstructBasis(BasisType&& basis)
{
	using DecayedBasisType = typename std::decay<BasisType>::type;
	using IteratorType = typename std::result_of<decltype(&DecayedBasisType::begin)(BasisType)>::type;
	return parallelConstructBasis(basis, 
			typename std::iterator_traits<IteratorType>::iterator_category());
}
