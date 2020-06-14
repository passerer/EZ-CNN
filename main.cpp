#include<iostream>
#include<vector>
#include <random>
#include"Tensor.h"
#include"ActivationLayer.h"
#include"ConvolutionLayer.h"
#include"PoolingLayer.h"
#include"SoftmaxLayer.h"
#include"LossLayer.h"
#include"FullyConnectLayer.h"

using namespace std;
int main()
{
	vector<unsigned int> inDim{ 10, 10};
	vector<unsigned int> preDim{ 10 };
	TensorXF in(inDim);
	Tensor<unsigned> pre(preDim);
	std::default_random_engine e;
	std::normal_distribution<float> n(0., 1.);
	
	for (int i = 0; i < in.size(); ++i)
	{
		in[i] = 0.1;
	//	cout << in[i] << "  ";
	}
	
	cout <<endl<< "input prepared" << endl;
	pre.fillData(3);
	FullyConnectLayer F1(10, 10, 10);
	FullyConnectLayer F2(10, 10, 10);
	SoftmaxLayer S(10, 10);
	CrossEntropyLossLayer L(10, 10);
	for (int i = 0; i < 20; ++i)
	{
		auto out1 = F1.forward(in);
		auto out2 = F2.forward(out1);
		auto out3 = S.forward(out2);
		L.forward(out3, pre);
		cout << L.getLoss() << endl;
		auto out4 = L.backward(out3, pre);
		auto out5 = S.backward(out4);
		auto out6 = F2.backward(out5);
		auto out7 = F1.backward(out6);
		F1.update();
		F2.update();
	}
	
	system("pause");
	return 0;
}

