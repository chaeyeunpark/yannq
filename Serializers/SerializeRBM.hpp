#ifndef CY_SERIALIZE_RBM_HPP
#define CY_SERIALIZE_RBM_HPP
#include <boost/serialization/complex.hpp>
#include "Machines/RBM.hpp"
namespace boost{
    namespace serialization
	{

        template<   class Archive, 
                    typename T>
		inline void save(
            Archive & ar, 
            const yannq::RBM<T, false>& g, 
            const unsigned int version)
            {
				int n = g.getN();
				int m = g.getM();

                ar & n;
                ar & m;
				ar & boost::serialization::make_array(g.getW().data(), m*n);
            }
        template<   class Archive, 
                    typename T>
		inline void load(
            Archive & ar, 
            yannq::RBM<T, false>& g, 
            const unsigned int version)
            {
				int n, m;

                ar & n;
                ar & m;

				typename yannq::RBM<T, true>::Matrix W(m,n);

				ar & boost::serialization::make_array(W.data(), m*n);

				g.resize(n,m);
				g.setW(W);
            }
        template<   class Archive, 
                    typename T>
        inline void save(
            Archive & ar, 
            const yannq::RBM<T, true>& g, 
            const unsigned int version)
            {
				int n = g.getN();
				int m = g.getM();

                ar & n;
                ar & m;
				ar & boost::serialization::make_array(g.getA().data(), n);
				ar & boost::serialization::make_array(g.getB().data(), m);
				ar & boost::serialization::make_array(g.getW().data(), m*n);
            }
        template<   class Archive, 
                    typename T>
		inline void load(
            Archive & ar, 
            yannq::RBM<T, true>& g, 
            const unsigned int version)
            {
				int n, m;

                ar & n;
                ar & m;

				typename yannq::RBM<T, true>::Vector A(n);
				typename yannq::RBM<T, true>::Vector B(m);
				typename yannq::RBM<T, true>::Matrix W(m,n);

				ar & boost::serialization::make_array(A.data(), n);
				ar & boost::serialization::make_array(B.data(), m);
				ar & boost::serialization::make_array(W.data(), m*n);

				g.resize(n,m);
				g.setA(A);
				g.setB(B);
				g.setW(W);
            }
        template<   class Archive, 
                    typename T,
					bool useBias>
        inline void serialize(
            Archive & ar, 
			yannq::RBM<T, useBias>& g,
            const unsigned int version)
        {
            split_free(ar, g, version);
        }

    } // namespace serialization
} // namespace boost
#endif//CY_SERIALIZE_RBM_HPP
