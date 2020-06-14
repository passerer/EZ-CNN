#pragma once

#include<vector>
#include "Layer.h"

class ReshapeLayer :public Layer
{
private:
	TensorXF  x;
	TensorXF dx;
public:
	ReshapeLayer(unsigned int nb, unsigned int nh ,unsigned int nw,unsigned int nc) :
		x(U{ nb, nh*nw*nc }, 0.f),
		dx(U{nb,nh,nw,nc},0.f)
	{ setLayerType(LayerType::Reshape); }
	TensorXF forward( TensorXF& input);
	TensorXF backward( TensorXF& input);
	void update();
};