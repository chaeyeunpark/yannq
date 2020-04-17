#pragma once
#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>

#define LAYER_ADD_SERIALIZER(TemplateClassName, TypeName) \
	CEREAL_REGISTER_TYPE(TemplateClassName<TypeName>); \
	CEREAL_REGISTER_POLYMORPHIC_RELATION( \
			yannq::AbstractLayer<TypeName>, TemplateClassName<TypeName>);

