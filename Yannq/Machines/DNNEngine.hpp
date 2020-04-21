#pragma once

#include <dnnl.hpp>

class DNNEngine
{
private:
	dnnl::engine engine_;

	DNNEngine()
		: engine_(dnnl::engine::kind::cpu, 0)
	{
	}
public:
	static dnnl::engine& getEngine()
	{
		static DNNEngine instance;
		return instance.engine_;
	}
};
