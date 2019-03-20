#ifndef CY_SERIALIZE_RBM_HPP
#define CY_SERIALIZE_RBM_HPP
#include <boost/serialization/complex.hpp>
#include "Machines/Jastrow.hpp"
namespace boost{
    namespace serialization
	{

        template<class Archive, typename T>
		inline void save(
            Archive & ar, 
            const nnqs::Jastrow<T>& g, 
            const unsigned int version)
		{
			int n = g.getN();

			ar & n;
			ar & boost::serialization::make_array(g.getA().data(), n);
			ar & boost::serialization::make_array(g.getJ().data(), n*n);
		}

        template<class Archive, typename T>
		inline void load(
            Archive & ar, 
            nnqs::Jastrow<T>& g, 
            const unsigned int version)
		{
			int n;

			ar & n;

			typename nnqs::Jastrow<T>::Vector a(n);
			typename nnqs::Jastrow<T>::Matrix J(n,n);

			ar & boost::serialization::make_array(a.data(), n);
			ar & boost::serialization::make_array(J.data(), n*n);
			
			g.resize(n);
			g.setA(a);
			g.setJ(J);
		}

        template<class Archive, typename T>
        inline void serialize(
            Archive & ar, 
			nnqs::Jastrow<T>& g,
            const unsigned int version)
        {
            split_free(ar, g, version);
        }

    } // namespace serialization
} // namespace boost
#endif//CY_SERIALIZE_RBM_HPP
