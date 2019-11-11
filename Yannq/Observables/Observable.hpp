#ifndef YANNQ_OBSERVABLES_OBSERVABLE_HPP
#define YANNQ_OBSERVABLES_OBSERVABLE_HPP

template<class Derived>
class Observable
{
private:

public:
	void initIter(int nsmp)
	{
		static_cast<Derived*>(this)->initIter(nsmp);
	}

	template<class Elt, class Sample>
	void eachSample(int n, Elt&& elt, Sample&& sample)
	{
		static_cast<Derived*>(this)->eachSample(n, elt,sample);
	}

	void finIter()
	{
		static_cast<Derived*>(this)->finIter();
	}
};
#endif//YANNQ_OBSERVABLES_OBSERVABLE_HPP
