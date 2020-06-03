#pragma once 
#include"Layer.h"

class PoolingLayer :public Layer
{
public:
	PoolingLayer(unsigned int _stride=2):stride(_stride){ setLayerType(LayerType::Pooling); }
protected:
	unsigned int stride;
};

class MaxPoolingLayer :public PoolingLayer
{
public:
	void forward( TensorXF& input, TensorXF& output);
	void backward(const TensorXF& input, const TensorXF& output,
		const TensorXF& preDiff, TensorXF& nextDiff);
	void update();
};

class AveragePoolingLayer :public PoolingLayer
{
public:
	void forward(TensorXF& input, TensorXF& output);
	void backward(const TensorXF& input, const TensorXF& output,
		const TensorXF& preDiff, TensorXF& nextDiff);
	void update();
};