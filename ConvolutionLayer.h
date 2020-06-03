#pragma once 
#include"Layer.h"

class ConvolutionLayer :public Layer
{
private:
	TensorXF weight;
	TensorXF bias;
public:
	ConvolutionLayer(){ setLayerType(LayerType::Convolution); }
	void init();
	void forward(const TensorXF& input, TensorXF& output);
	void backward(const TensorXF& input, const TensorXF& output,
		const TensorXF& preDiff, TensorXF& nextDiff);
	void update();
};

