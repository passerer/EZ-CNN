#pragma once 
#include"Layer.h"

class FullyConnectLayer :public Layer
{
private:
	TensorXF weight;
	TensorXF bias;
public:
	FullyConnectLayer(unsigned int ki,unsigned int ko):
		weight(std::vector<unsigned int>{ki,ko}),
		bias(std::vector<unsigned int>{ko})
	{ setLayerType(LayerType::FullConnect); 
	  init();
	}
	void init();
	void forward( TensorXF& input, TensorXF& output);
	void backward(const TensorXF& input, const TensorXF& output,
		const TensorXF& preDiff, TensorXF& nextDiff);
	void update();
};