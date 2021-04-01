#include <tbb/tbb.h>


int numThreads(int numThreads = -1)
{
	char* var = std::getenv("TBB_NUM_THREADS");
	int val;
	if((var != nullptr) && (var[0] != '\0') && 
			(sscanf(var, "%d", &val) == 1) && (val > 0) &&
			(numThreads == -1))
	{
		numThreads = val;
	}
	else
	{
		numThreads = tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism);
	}
	return numThreads;
}
