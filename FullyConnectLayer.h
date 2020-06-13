#pragma once 
#include"Layer.h"

class FullyConnectLayer :public Layer
{
private:
	TensorXF weight;
	TensorXF bias;
	TensorXF dx;
	TensorXF dw;
	TensorXF db;
	TensorXF preInput;
public:
	FullyConnectLayer(unsigned int nb,unsigned int ni,unsigned int no):
		weight(U{ ni, no},0.f),
		bias(U{ no }, 0.f),
		dx(U{ nb, ni }, 0.f),
		dw(U{ ni, no }, 0.f),
		db(U{ no }, 0.f),
		preInput(U{ nb, ni},0.f)
	{ setLayerType(LayerType::FullConnect); 
	  init();
	}
	void init();
	TensorXF forward( TensorXF& input);
	TensorXF backward(TensorXF& input);
	void update();
};