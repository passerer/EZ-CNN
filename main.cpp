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
	vector<float> tmp1;
	vector<unsigned int> tmp2;
	for (int i = 1; i < 101; ++i)
	{
		tmp1.push_back((i/10) / 100.f);
	}
	for (int i = 0; i < 10; ++i)
	{
		tmp2.push_back(i/5);
	}
	in = tmp1;
	pre = tmp2;
	cout <<endl<< "input prepared" << endl;
	ReluLayer R (10, 10);
	FullyConnectLayer F1(10, 10, 10);
	FullyConnectLayer F2(10, 10, 10);
	SoftmaxLayer S(10, 10);
	CrossEntropyLossLayer L(10, 10);
	for (int i = 0; i < 50; ++i)
	{
		auto out1 = F1.forward(in);
		auto out2 = R.forward(out1);
		auto out3 = F2.forward(out2);
		auto out4 = S.forward(out3);
		L.forward(out4, pre);
		cout << L.getLoss() << endl;
		auto out5 = L.backward(out4, pre);
		auto out6 = S.backward(out5);
		auto out7 = F2.backward(out6);
		auto out8 = R.backward(out7);
		auto out9 = F1.backward(out8);
		F1.update();
		F2.update();
	}
	
	system("pause");
	return 0;
}

