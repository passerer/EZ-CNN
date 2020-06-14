#pragma once 
#include"Layer.h"

class ConvolutionLayer :public Layer
{
private:
	TensorXF weight;
	TensorXF bias;
	TensorXF dw;
	TensorXF db;
	TensorXF preInput;
public:
	ConvolutionLayer(unsigned int nb,unsigned int nh,unsigned int nw,unsigned int c, unsigned int nc,unsigned int h=3,unsigned int w=3):
		weight(U{h, w, c, nc}),
		bias(U{nc}), 
		dw(U{h, w, c, nc},0.f),
		db(U{ nc }, 0.f),
		preInput(U{nb,nh,nw,c,nc},0.f)
	{
		setLayerType(LayerType::Convolution);
		init();
	}
	void init();
	TensorXF forward(TensorXF& input);
	TensorXF backward(TensorXF& input);
	void update();
};

