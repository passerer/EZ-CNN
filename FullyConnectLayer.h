#pragma once 
#include"Layer.h"

class FullyLayer :public Layer
{
private:
	TensorXF weight;
	TensorXF bias;
public:
	FullyLayer(){ setLayerType(LayerType::FullConnect); }
	void init();
	void forward(const TensorXF& input, TensorXF& output);
	void backward(const TensorXF& input, const TensorXF& output,
		const TensorXF& preDiff, TensorXF& nextDiff);
	void update();
};