#pragma once

#define GCC_VERSION (__GNUC__ * 10000 \
		                     + __GNUC_MINOR__ * 100 \
		                     + __GNUC_PATCHLEVEL__)

#if GCC_VERSION < 80100 
#include  <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include  <filesystem>
namespace fs = std::filesystem;
#endif
