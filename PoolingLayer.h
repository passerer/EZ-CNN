#pragma once 
#include"Layer.h"

class PoolingLayer :public Layer
{
public:
	PoolingLayer(){ setLayerType(LayerType::Pooling); }
};

class MaxPoolingLayer :public PoolingLayer
{
public:
	void forward(const TensorXF& input, TensorXF& output);
	void backward(const TensorXF& input, const TensorXF& output,
		const TensorXF& preDiff, TensorXF& nextDiff);
	void update();
};

class AveragePoolingLayer :public PoolingLayer
{
public:
	void forward(const TensorXF& input, TensorXF& output);
	void backward(const TensorXF& input, const TensorXF& output,
		const TensorXF& preDiff, TensorXF& nextDiff);
	void update();
};