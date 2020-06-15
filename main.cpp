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
#include"ReshapeLayer.h"

#include"minist.h"

using namespace std;
int main()
{
	vector<unsigned int> inDim{ 20, 28,28,1};
	vector<unsigned int> preDim{ 20 };
	TensorXF in(inDim);
	Tensor<unsigned int> label(preDim);
	in = train_data;
	label = train_label;
	
	cout <<endl<< "input prepared" << endl;

	ConvolutionLayer conv1  (20, 28, 28, 1, 8);
	ReluLayer relu1(20, 28, 28, 8);
	MaxPoolingLayer max1(20, 28, 28, 8);
	ConvolutionLayer conv2(20, 14, 14, 8, 1);
	ReluLayer relu2(20, 14, 14, 1);
	MaxPoolingLayer max2(20, 14, 14, 1);
	ReshapeLayer reshape(20, 7, 7, 1);
	FullyConnectLayer F1(20, 7*7, 10);
	SoftmaxLayer S(20, 10);
	CrossEntropyLossLayer L(20, 10);
	for (int i = 0; i < 10;++i)
	{
	 //forward
		auto out1 = conv1.forward(in);
		auto out2 = relu1.forward(out1);
		auto out3 = max1.forward(out2);
		auto out4 = conv2.forward(out3);
		auto out5 = relu2.forward(out4);
		auto out6 = max2.forward(out5);
		auto out7 = reshape.forward(out6);
		auto out8 = F1.forward(out7);
		auto out9 = S.forward(out8);
		auto loss = L.forward(out9, label);
		cout << loss << endl;
	
		//backward
		auto out10 = L.backward(out9, label);
		auto out11 = S.backward(out10);
		auto out12 = F1.backward(out11);
		auto out13 = reshape.backward(out12);
		auto out14 = max2.backward(out13);
		auto out15 = relu2.backward(out14);
		auto out16 = conv2.backward(out15);
		auto out17 = max1.backward(out16);
		auto out18 = relu1.backward(out17);
		auto out19 = conv1.backward(out18);

		//update
		F1.update();
		conv2.update();
		conv1.update();
	}

	/*
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
	*/
	system("pause");
	return 0;
}

