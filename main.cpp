#include <iostream>
#include <vector>
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

#define BATCH  10
#define CLASS 10
#define TEST_OUTPUT 10
using namespace std;
int main()
{
	// prepare input 
	vector<unsigned int> inDim{ BATCH, 12,12,1 };
	vector<unsigned int> preDim{ BATCH };
	TensorXF in(inDim);
	Tensor<unsigned int> label(preDim);

	std::random_device rd;
	std::default_random_engine e{rd()};
	std::normal_distribution<float> n(0., 0.3/CLASS );
	for (unsigned int i = 0; i < BATCH; i++)
	{
		label(vector<unsigned int> {i}) = i / (BATCH/CLASS);
	}
	cout <<endl<< "input prepared" << endl;
	
	// build network structure
//	ConvolutionLayer conv1(BATCH, 12, 12, 1, 4);
//	ReluLayer relu1(BATCH, 12, 12, 4);
//	MaxPoolingLayer max1(BATCH, 12, 12, 4);
//	ConvolutionLayer conv2(BATCH, 6, 6, 4, 8);
//	ReluLayer relu2(BATCH, 6, 6, 8);
//	MaxPoolingLayer max2(BATCH, 6, 6,8);
	ReshapeLayer reshape(BATCH, 12, 12, 1);
	FullyConnectLayer F1(BATCH, 12*12*1, CLASS * 50);
	ReluLayer relu3(BATCH, CLASS*50);
	FullyConnectLayer F2(BATCH, CLASS * 50, CLASS*10);
	ReluLayer relu4(BATCH, CLASS*10);
	FullyConnectLayer F3(BATCH, CLASS * 10, CLASS);
	ReluLayer relu5(BATCH, CLASS);
	SoftmaxLayer S(BATCH, CLASS);
	CrossEntropyLossLayer L(BATCH, CLASS);
	

	//start to train network
	for (int epoch = 0; epoch < 80; ++epoch)
	{
		cout << endl << "------------------------------------------" << endl;
		//forward
		for (unsigned int i = 0; i < BATCH * 12 * 12 * 1; i++)
		{
			in[i] = (float)(i / (12 * 12 * 1))/BATCH*2 -1. +n(e);
		}
	//	auto out1 = conv1.forward(in);
	//	auto out2 = relu1.forward(out1);
	//	auto out3 = max1.forward(out2);
	//	auto out4 = conv2.forward(out3);
	//	auto out5 = relu2.forward(out4);
	//	auto out6 = max2.forward(out5);
		auto out7 = reshape.forward(in);
		auto out8 = F1.forward(out7);
		auto out9 = relu3.forward(out8);
		auto out10 = F2.forward(out9);
		auto out11 = relu4.forward(out10);
		auto out12 = F3.forward(out11);
		auto out13 = relu5.forward(out12);
		auto out14 = S.forward(out13);
		cout << endl << "softmax" << endl;
		for (unsigned int i = 0; i < BATCH; i++)
		{
			cout << out14(vector<unsigned int> {i, label(vector<unsigned int>{i})}) << "  ";
		}
		auto loss = L.forward(out14, label);
		cout <<endl<< "loss:"<< loss << endl;

		//backward
		auto out15 = L.backward(out14, label);
		cout << endl << "loss backward" << endl;
		for (unsigned int i = 0; i < BATCH; i++)
		{
			cout << out15(vector<unsigned int> {i, label(vector<unsigned int>{i})}) << "  ";
		}
		auto out16 = S.backward(out15);
		auto out17 = relu5.backward(out16);
		auto out18 = F3.backward(out17);
		auto out19 = relu4.backward(out18);
		auto out20 = F2.backward(out19);
		auto out21 = relu3.backward(out20);
		auto out22 = F1.backward(out21);
		auto out23 = reshape.backward(out22);
//		auto out20 = max2.backward(out19);
//		auto out21 = relu2.backward(out20);
//		auto out22 = conv2.backward(out21);
//		auto out23 = max1.backward(out22);
//		auto out24 = relu1.backward(out23);
//		auto out25 = conv1.backward(out24);

		//update
		F3.update();
		F2.update();
		F1.update();
//		conv2.update();
//		conv1.update();
	}
	
	system("pause");
	return 0;
}

