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
	// prepare input 
	vector<unsigned int> inDim{ 2, 28,28,1};
	vector<unsigned int> preDim{ 2 };
	TensorXF in(inDim);
	Tensor<unsigned int> label(preDim);
	
	cout <<endl<< "input prepared" << endl;

	// build network structure
	ConvolutionLayer conv1  (2, 28, 28, 1, 4);
	ReluLayer relu1(2, 28, 28, 4);
	MaxPoolingLayer max1(2, 28, 28, 4);
	ConvolutionLayer conv2(2, 14, 14, 4, 8);
	ReluLayer relu2(2, 14, 14, 8);
	MaxPoolingLayer max2(2, 14, 14, 8);
	ReshapeLayer reshape(2, 7, 7, 8);
	FullyConnectLayer F1(2, 7*7*8, 100);
	ReluLayer relu3(2, 100);
	FullyConnectLayer F2(2, 100, 10);
	SigmoidLayer sigmoid(2, 10);
	SoftmaxLayer S(2, 10);
	CrossEntropyLossLayer L(2, 10);
	//start to train network
	for (int epoch = 0; epoch < 10; ++epoch)
	{
		for (int batch = 0; batch < 10; batch++)
		{
			//forward
			in = vector<float>(train_data.begin() + batch * 2 * 28 * 28, train_data.begin() + (batch + 1) * 2 * 28 * 28);
			label = vector<unsigned int>(train_label.begin() + batch * 2, train_label.begin() + (batch + 1) * 2);
			auto out1 = conv1.forward(in);
			auto out2 = relu1.forward(out1);
			auto out3 = max1.forward(out2);
			auto out4 = conv2.forward(out3);
			auto out5 = relu2.forward(out4);
			auto out6 = max2.forward(out5);
			auto out7 = reshape.forward(out6);
			auto out8 = F1.forward(out7);
			auto out9 = relu3.forward(out8);
			auto out10 = F2.forward(out9);
			auto out11 = sigmoid.forward(out10);
			auto out12 = S.forward(out11);

			auto loss = L.forward(out12, label);
			cout << loss << endl;

			//backward
			auto out13 = L.backward(out12, label);
			auto out14 = S.backward(out13);
			auto out15 = sigmoid.backward(out14);
			auto out16 = F2.backward(out15);
			auto out17 = relu3.backward(out16);
			auto out18 = F1.backward(out17);
			auto out19 = reshape.backward(out18);
			auto out20 = max2.backward(out19);
			auto out21 = relu2.backward(out20);
			auto out22 = conv2.backward(out21);
			auto out23 = max1.backward(out22);
			auto out24 = relu1.backward(out23);
			auto out25 = conv1.backward(out24);

			//update
			F2.update();
			F1.update();
			conv2.update();
			conv1.update();
		}
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

