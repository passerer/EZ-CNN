#pragma once 
#include"Layer.h"

class ConvolutionLayer :public Layer
{
private:
	TensorXF weight;
	TensorXF bias;
public:
	ConvolutionLayer(unsigned int ki, unsigned int ko,unsigned int h=3,unsigned int w=3):
		weight(std::vector<unsigned int>{h,w,ki,ko}),
		bias(std::vector<unsigned int>{ko})
	{
		setLayerType(LayerType::Convolution);
		init();
	}
	void init();
	void forward(TensorXF& input, TensorXF& output);
	void backward(const TensorXF& input, const TensorXF& output,
		const TensorXF& preDiff, TensorXF& nextDiff);
	void update();
};

