#pragma once 
#include"Layer.h"

class PoolingLayer :public Layer
{
public:
	PoolingLayer(unsigned int nb, unsigned int nh, unsigned int nw, unsigned int nc, unsigned int _stride = 2)
		:stride(_stride), pos(U{ nb, nh,nw,nc }, 0), output(U{ nb, nh/_stride, nw/_stride, nc }, 0.f)
	{ setLayerType(LayerType::Pooling); }
protected:
	unsigned int stride;
	TensorXF pos;
	TensorXF output;
};

class MaxPoolingLayer :public PoolingLayer
{
public:
	TensorXF forward( TensorXF& input);
	TensorXF backward(TensorXF& input);
	void update();
};

class AveragePoolingLayer :public PoolingLayer
{
public:
	TensorXF forward(TensorXF& input);
	TensorXF backward(TensorXF& input);
	void update();
};