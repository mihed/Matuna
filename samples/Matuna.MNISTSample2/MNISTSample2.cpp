/*
* MNISTSample1.cpp
*
*  Created on: Jun 4, 2015
*      Author: Mikael
*/

#include "MNISTAssetLoader.h"

#include "Matuna.OCLHelper/OCLHelper.h"
#include "Matuna.OCLConvNet/OCLConvNet.h"
#include "Matuna.OCLConvNet/PerceptronLayer.h"
#include "Matuna.ConvNet/GradientDescentConfig.h"
#include "Matuna.ConvNet/PerceptronLayerConfig.h"
#include "Matuna.ConvNet/StandardOutputLayerConfig.h"
#include "Matuna.ConvNet/ConvolutionLayerConfig.h"
#include "Matuna.ConvNet/VanillaSamplingLayerConfig.h"
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
	size_t counter;

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

	virtual void UnmapInputAndTarget(T*, T*, int) override
	{

	}

	virtual void BatchFinished(T) override
	{
		//cout << "Counter: " << counter << endl;
	}

	virtual void EpochFinished() override
	{
		size_t correctClassifications = 0;
		size_t totalClassifications = tests.size();
		for (size_t i = 0; i < tests.size(); i++)
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

		printf("Correct classifications: %i \n", static_cast<int>(correctClassifications));
		printf("False classifications: %i \n", static_cast<int>(totalClassifications - correctClassifications));
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


//This represents the network given in http://research.microsoft.com/pubs/68920/icdar03.pdf
int main(int, char**)
{

	auto trainingImages = MNISTAssetLoader<float>::ReadTrainingImages();

	//Let us add a border of one too the images according to the paper
	vector<Matrix<float>> fixedTrainingImages;
	for (auto image : trainingImages)
		fixedTrainingImages.push_back(image.AddZeroBorder(0, 1, 0, 1));

	trainingImages.clear();

	auto testImages = MNISTAssetLoader<float>::ReadTestImages();
	vector<Matrix<float>> fixedTestImages;
	for (auto image : testImages)
		fixedTestImages.push_back(image.AddZeroBorder(0, 1, 0, 1));

	testImages.clear();

	auto testTargets = MNISTAssetLoader<float>::ReadTestTargets();
	auto trainingTargets = MNISTAssetLoader<float>::ReadTrainingTargets();
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
	inputDesc.Height = fixedTrainingImages[0].RowCount();
	inputDesc.Width = fixedTrainingImages[0].ColumnCount();
	inputDesc.Units = 1;
	inputDataDescriptions.push_back(inputDesc);
	unique_ptr<ConvNetConfig> config(new ConvNetConfig(inputDataDescriptions));
	vector<OCLDeviceInfo> deviceInfos;
	deviceInfos.push_back(deviceInfo);

	unique_ptr<ConvolutionLayerConfig> convLayerConfig1(new ConvolutionLayerConfig(5, 5, 5, MatunaTanhActivation));
	unique_ptr<VanillaSamplingLayerConfig> vanillaLayerConfig1(new VanillaSamplingLayerConfig(2, 2));
	unique_ptr<ConvolutionLayerConfig> convLayerConfig2(new ConvolutionLayerConfig(50, 5, 5, MatunaTanhActivation));
	unique_ptr<VanillaSamplingLayerConfig> vanillaLayerConfig2(new VanillaSamplingLayerConfig(2, 2));
	unique_ptr<PerceptronLayerConfig> perceptronConfig1(new PerceptronLayerConfig(100, MatunaTanhActivation));
	unique_ptr<PerceptronLayerConfig> perceptronConfig2(new PerceptronLayerConfig(10, MatunaSoftMaxActivation));
	unique_ptr<StandardOutputLayerConfig> outputLayerConfig(new StandardOutputLayerConfig(MatunaCrossEntropy));
	config->AddToBack(move(convLayerConfig1));
	config->AddToBack(move(vanillaLayerConfig1));
	config->AddToBack(move(convLayerConfig2));
	config->AddToBack(move(vanillaLayerConfig2));
	config->AddToBack(move(perceptronConfig1));
	config->AddToBack(move(perceptronConfig2));
	config->SetOutputConfig(move(outputLayerConfig));
	OCLConvNet<float> network(deviceInfos, move(config));

	auto tempTrainer = new TestConvNetTrainer<float>(network.InputForwardDataDescriptions(), network.OutputForwardDataDescriptions(),
		network.InputForwardMemoryDescriptions(),
		network.OutputForwardMemoryDescriptions(), &network);
	tempTrainer->SetInputs(fixedTrainingImages);
	tempTrainer->SetTargets(trainingTargets);
	tempTrainer->SetTests(fixedTestImages);
	tempTrainer->SetTestTargets(testTargets);

	unique_ptr<ConvNetTrainer<float>> trainer(tempTrainer);

	unique_ptr<GradientDescentConfig<float>> trainingConfig(new GradientDescentConfig<float>());
	trainingConfig->SetBatchSize(60);
	trainingConfig->SetEpochs(40);
	auto callBack = [] (int epoch) 
	{ 
		if (epoch < 4)
			return 0.001f;
		else if (epoch < 10)
			return 0.0001f;
		else
			return 0.000001f;
	};

	trainingConfig->SetStepSizeCallback(callBack);
	trainingConfig->SetSamplesPerEpoch(60000);

	network.TrainNetwork(move(trainer), move(trainingConfig));

	return 0;
}



