/*
* MNISTSample1.cpp
*
*  Created on: Jun 4, 2015
*      Author: Mikael
*/

#include "AssetLoader.h"

#include "Matuna.OCLHelper/OCLHelper.h"
#include "Matuna.OCLConvNet/OCLConvNet.h"
#include "Matuna.OCLConvNet/PerceptronLayer.h"
#include "Matuna.ConvNet/GradientDescentConfig.h"
#include "Matuna.ConvNet/PerceptronLayerConfig.h"
#include "Matuna.ConvNet/StandardOutputLayerConfig.h"
#include "Matuna.ConvNet/ConvolutionLayerConfig.h"
#include "Matuna.ConvNet/ConvNetTrainer.h"
#include "Matuna.ConvNet/GradientDescentConfig.h"

#include <stdio.h>
#include <iostream>
#include <string>

using namespace std;
using namespace Matuna::MachineLearning;
using namespace Matuna::Math;
using namespace Matuna::Helper;

template<class T> 
class TestConvNetTrainer: public ConvNetTrainer<T>
{
private:
	OCLConvNet<T>* network;
	vector<Matrix<T>> inputs;
	vector<Matrix<T>> targets;
	vector<Matrix<T>> tests;
	vector<Matrix<T>> testTargets;
	int counter;

public:
	TestConvNetTrainer( const vector<LayerDataDescription>& inputDataDescriptions,
		const vector<LayerDataDescription>& targetDataDescriptions,
		const vector<LayerMemoryDescription>& inputMemoryDescriptions,
		const vector<LayerMemoryDescription>& targetMemoryDescriptions, 
		OCLConvNet<T>* network) : ConvNetTrainer<T>(inputDataDescriptions, targetDataDescriptions, inputMemoryDescriptions, targetMemoryDescriptions)
	{
		counter = 0;
		this->network = network;
	}

	~TestConvNetTrainer()
	{

	}

	virtual void MapInputAndTarget(T*& input, T*& target,int& formatIndex) override
	{

		input = inputs[counter].Data;
		target = targets[counter].Data;
		formatIndex = 0;

		counter++;
		if (counter >= inputs.size())
			counter = 0;
	}

	virtual void UnmapInputAndTarget(T* input, T* target,int formatIndex) override
	{

	}

	virtual void BatchFinished(T error) override
	{
		//cout << "Counter: " << counter << endl;
	}

	virtual void EpochFinished() override
	{
		T totalError = 0;
		int correctClassifications = 0;
		int totalClassifications = tests.size();
		for (int i = 0; i < tests.size(); i++)
		{
			Matrix<T> result(10, 1, network->FeedForwardAligned(tests[i].Data, 0).get());
			int maxIndex = 0;
			T maxValue = 0;
			for (int k = 0; k < 10; k++)
			{
				if (result.At(k, 0) > maxValue)
				{
					maxValue = result.At(k, 0);
					maxIndex = k;
				}
			}

			if (testTargets[i].At(maxIndex, 0) != 0)
				correctClassifications++;
		}

		printf("Correct classifications: %i \n", correctClassifications);
		printf("False classifications: %i \n", totalClassifications - correctClassifications);
		printf("Performance: %f \n", float(correctClassifications) / totalClassifications);
	}

	virtual void EpochStarted() override
	{
		cout << "Epoch started" << endl;
	}

	virtual void BatchStarted() override
	{
		//cout << "Batch started" << endl;
	}

	void SetTests(vector<Matrix<T>> tests)
	{
		this->tests = tests;
	}

	void SetTestTargets(vector<Matrix<T>> testTargets)
	{
		this->testTargets = testTargets;
	}

	void SetInputs(vector<Matrix<T>> inputs)
	{
		this->inputs = inputs;
	}

	void SetTargets(vector<Matrix<T>> targets)
	{
		this->targets = targets;
	}
};


int main(int argc, char* argv[])
{

	auto trainingImages = AssetLoader<float>::ReadTrainingImages();
	auto testImages = AssetLoader<float>::ReadTestImages(1000);
	auto testTargets = AssetLoader<float>::ReadTestTargets(1000);
	auto trainingTargets = AssetLoader<float>::ReadTrainingTargets();
	auto platformInfos = OCLHelper::GetPlatformInfos();

	if (platformInfos.size() == 0)
	{
		cout << "No available OCL platforms" << endl;
		return 1;
	}

	//If there's an nvidia gpu available, use it. Otherwise, train on the first best device
	auto deviceInfo = OCLHelper::GetDeviceInfos(platformInfos[0])[0];
	for (auto& platformInfo : platformInfos)
	{
		if (platformInfo.PlatformName().find("NVIDIA") != string::npos)
		{
			deviceInfo = OCLHelper::GetDeviceInfos(platformInfo)[0];
			break;
		}
	}

	vector<LayerDataDescription> inputDataDescriptions;
	LayerDataDescription inputDesc;
	inputDesc.Height = trainingImages[0].RowCount();
	inputDesc.Width = trainingImages[0].ColumnCount();
	inputDesc.Units = 1;
	inputDataDescriptions.push_back(inputDesc);
	unique_ptr<ConvNetConfig> config(new ConvNetConfig(inputDataDescriptions));
	vector<OCLDeviceInfo> deviceInfos;
	deviceInfos.push_back(deviceInfo);

	unique_ptr<ConvolutionLayerConfig> convLayerConfig1(new ConvolutionLayerConfig(16, 8, 8, MatunaTanhActivation));
	unique_ptr<ConvolutionLayerConfig> convLayerConfig2(new ConvolutionLayerConfig(32, 4, 4, MatunaTanhActivation));
	unique_ptr<PerceptronLayerConfig> perceptronConfig1(new PerceptronLayerConfig(64, MatunaTanhActivation));
	unique_ptr<PerceptronLayerConfig> perceptronConfig2(new PerceptronLayerConfig(10, MatunaSoftMaxActivation));
	unique_ptr<StandardOutputLayerConfig> outputLayerConfig(new StandardOutputLayerConfig(MatunaCrossEntropy));
	config->AddToBack(move(convLayerConfig1));
	config->AddToBack(move(convLayerConfig2));
	config->AddToBack(move(perceptronConfig1));
	config->AddToBack(move(perceptronConfig2));
	config->SetOutputConfig(move(outputLayerConfig));
	OCLConvNet<float> network(deviceInfos, move(config));

	auto tempTrainer = new TestConvNetTrainer<float>(network.InputForwardDataDescriptions(), network.OutputForwardDataDescriptions(),
		network.InputForwardMemoryDescriptions(),
		network.OutputForwardMemoryDescriptions(), &network);
	tempTrainer->SetInputs(trainingImages);
	tempTrainer->SetTargets(trainingTargets);
	tempTrainer->SetTests(testImages);
	tempTrainer->SetTestTargets(testTargets);

	unique_ptr<ConvNetTrainer<float>> trainer(tempTrainer);

	unique_ptr<GradientDescentConfig<float>> trainingConfig(new GradientDescentConfig<float>());
	trainingConfig->SetBatchSize(60);
	trainingConfig->SetEpochs(10);
	auto callBack = [] (int x) 
	{ 
		return 0.001;
	};

	trainingConfig->SetStepSizeCallback(callBack);
	trainingConfig->SetSamplesPerEpoch(60000);

	network.TrainNetwork(move(trainer), move(trainingConfig));

	return 0;
}



