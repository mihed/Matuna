/*
* OCLConvNetBackMaxPoolingTest.cpp
*
*  Created on: Jun 23, 2015
*      Author: Mikael
*/

#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"
#include "Matuna.OCLHelper/OCLHelper.h"
#include "Matuna.OCLConvNet/OCLConvNet.h"
#include "Matuna.OCLConvNet/ConvolutionLayer.h"
#include "Matuna.ConvNet/ConvolutionLayerConfig.h"
#include "Matuna.ConvNet/StandardOutputLayerConfig.h"
#include "Matuna.ConvNet/MaxPoolingLayerConfig.h"
#include "Matuna.OCLConvNet/MaxPoolingLayer.h"
#include "Matuna.Math/Matrix.h"

#include <memory>
#include <cmath>
#include <random>
#include <tuple>
#include <type_traits>

using namespace Matuna::MachineLearning;
using namespace Matuna::Helper;
using namespace Matuna::Math;



unique_ptr<ConvNetConfig> CreateRandomConvNetMaxPoolingConfig(mt19937& mt,
															  uniform_int_distribution<int>& layerGenerator,
															  uniform_int_distribution<int>& dimensionGenerator,
															  uniform_int_distribution<int>& unitGenerator,
															  uniform_int_distribution<int>& samplingSizeGenerator,
															  bool useDivisableLayers)
{

	cout << "\n\n-------------------Network-------------------" << endl;

	int layerCount = layerGenerator(mt);

	INFO("Creating the layers config");
	int requiredWidthMultiple = 1;
	int requiredHeightMultiple = 1;
	vector<MaxPoolingLayerConfig*> maxConfigs;
	for (int i = 0; i < layerCount; i++)
	{
		cout << "----------Layer " << i << "----------------" << endl;
		int widthSamplingSize = samplingSizeGenerator(mt);
		requiredWidthMultiple *= widthSamplingSize;
		int heightSamplingSize = samplingSizeGenerator(mt);
		requiredHeightMultiple *= heightSamplingSize;
		maxConfigs.push_back(new MaxPoolingLayerConfig(widthSamplingSize, heightSamplingSize));

		cout << "Width Sampling: " << widthSamplingSize << " Height Sampling: " << heightSamplingSize << endl;
	}

	//Let us now find the dimension that is closest to the multiples
	int width = dimensionGenerator(mt);
	int height = dimensionGenerator(mt);
	int units = unitGenerator(mt);

	if (useDivisableLayers)
	{
		width = static_cast<int>(ceil(double(width) / requiredWidthMultiple) * requiredWidthMultiple);
		height = static_cast<int>(ceil(double(height) / requiredHeightMultiple) * requiredHeightMultiple);
		CHECK((width % requiredWidthMultiple) == 0);
		CHECK((height % requiredHeightMultiple) == 0);
	}

	vector<LayerDataDescription> dataDescriptions;
	LayerDataDescription dataDescription;

	dataDescription.Width = width;
	dataDescription.Height = height;
	dataDescription.Units = units;

	cout << endl << "Width: " << width << " Height: " << height << " Units: " << units << endl << endl;

	dataDescriptions.push_back(dataDescription);
	unique_ptr<ConvNetConfig> config(new ConvNetConfig(dataDescriptions));

	for(auto maxConfig : maxConfigs)
		config->AddToBack(unique_ptr<ForwardBackPropLayerConfig>(maxConfig));

	unique_ptr<StandardOutputLayerConfig> outputConfig(new StandardOutputLayerConfig());
	config->SetOutputConfig(move(outputConfig));

	return move(config);
}


SCENARIO("Creating a max sampling network")
{
	auto platformInfos = OCLHelper::GetPlatformInfos();
	random_device device;
	mt19937 mt(device());
	uniform_int_distribution<int> dimensionGenerator(1, 1000);
	uniform_int_distribution<int> samplingSizeGenerator(1, 5);
	uniform_int_distribution<int> unitGenerator(1, 15);
	uniform_int_distribution<int> layerGenerator(1, 5);

	for (int dummy = 0; dummy < 50; dummy++)
	{

		vector<vector<OCLDeviceInfo>> deviceInfos;
		for (auto platformInfo : platformInfos)
			deviceInfos.push_back(OCLHelper::GetDeviceInfos(platformInfo));

		for (auto& deviceInfo : deviceInfos)
		{

			unique_ptr<ConvNetConfig> config;
			if (dummy > 24)
				config = CreateRandomConvNetMaxPoolingConfig(mt, layerGenerator, dimensionGenerator, unitGenerator, samplingSizeGenerator, true);
			else
				config = CreateRandomConvNetMaxPoolingConfig(mt, layerGenerator, dimensionGenerator, unitGenerator, samplingSizeGenerator, false);

			OCLConvNet<float> network(deviceInfo, move(config));

			auto layers = network.GetLayers();
			size_t layersCount= layers.size(); 
			vector<tuple<int, int>> samplingSizes;
			for (auto layer : layers)
			{
				auto config = dynamic_cast<MaxPoolingLayer<float>*>(layer)->GetConfig();
				samplingSizes.push_back(make_tuple(config.SamplingSizeWidth(), config.SamplingSizeHeight()));
			}

			LayerDataDescription inForwardDesc = network.InputForwardDataDescriptions()[0];
			LayerDataDescription outForwardDesc = network.OutputForwardDataDescriptions()[0];

			printf("Out width: %i, Out height: %i, Out units: %i \n", outForwardDesc.Width, outForwardDesc.Height, outForwardDesc.Units);

			vector<Matrixf> inputMatrices;
			for (int i = 0; i < inForwardDesc.Units; i++)
				inputMatrices.push_back(Matrixf::RandomNormal(inForwardDesc.Height, inForwardDesc.Width));

			unique_ptr<float[]> contiguousMemory(new float[inForwardDesc.TotalUnits()]);
			for (int i = 0; i < inForwardDesc.Units; i++)
				memcpy(contiguousMemory.get() + i * inForwardDesc.Width * inForwardDesc.Height, inputMatrices[i].Data, inForwardDesc.Width * inForwardDesc.Height * sizeof(float));

			INFO("Feed forwarding the max network");
			auto networkResult = network.FeedForwardUnaligned(contiguousMemory.get(), 0);

			INFO("Calculating the manual network");
			vector<tuple<int, int>> heightWidths;
			vector<vector<vector<tuple<int, int>>>> downSampleIndices;
			for (size_t i = 0; i < layersCount; i++)
			{
				auto& samplingSize = samplingSizes[i];
				vector<Matrixf> resultMatrices;
				vector<vector<tuple<int, int>>> downSampleIndicesForLayer;
				for (size_t j = 0; j < inputMatrices.size(); j++)
				{
					vector<tuple<int, int>> indicesForUnit;
					resultMatrices.push_back(inputMatrices[j].MaxDownSample(get<0>(samplingSize), get<1>(samplingSize), indicesForUnit));

					//if (i == 1)
					//	for (auto tempTest : indicesForUnit)
					//		printf("(%i, %i) \n", get<0>(tempTest), get<1>(tempTest));

					downSampleIndicesForLayer.push_back(indicesForUnit);
				}

				downSampleIndices.push_back(downSampleIndicesForLayer);

				auto tempMatrix = resultMatrices[0];
				heightWidths.push_back(make_tuple(tempMatrix.RowCount(), tempMatrix.ColumnCount()));
				inputMatrices = resultMatrices;
			}

			for(size_t i = 0; i < inputMatrices.size(); i++)
			{
				auto inputMatrix = inputMatrices[i];
				Matrixf networkMatrix(inputMatrix.RowCount(), inputMatrix.ColumnCount(), networkResult.get() + i * inputMatrix.RowCount() * inputMatrix.ColumnCount());
				auto difference = (inputMatrix - networkMatrix).Norm2Square();
				cout << "difference: " << difference << endl;
				CHECK(difference < 1E-13f);
			}

			INFO("Creating some random targets");
			vector<Matrixf> targets;
			for (int i = 0; i < outForwardDesc.Units; i++)
				targets.push_back(Matrixf::RandomNormal(outForwardDesc.Height, outForwardDesc.Width));

			unique_ptr<float[]> contiguousOutMemory(new float[outForwardDesc.TotalUnits()]);
			for (int i = 0; i < outForwardDesc.Units; i++)
				memcpy(contiguousOutMemory.get() + i * outForwardDesc.Width * outForwardDesc.Height, targets[i].Data, outForwardDesc.Width * outForwardDesc.Height * sizeof(float));

			INFO("Back propagating the max pool network");
			auto backNetworkResult = network.BackPropUnaligned(contiguousMemory.get(), 0, contiguousOutMemory.get());


			CHECK(inputMatrices.size() == targets.size());

			vector<Matrixf> backMatrices;
			INFO("Calculating the output layer here, it must be linear activation since we use max sampling");
			for (size_t i = 0; i < inputMatrices.size(); i++)
				backMatrices.push_back(inputMatrices[i] - targets[i]);

			for (int i = static_cast<int>(layersCount) - 1; i > 0; i--)
			{
				auto& heigthWidth = heightWidths[i - 1];
				vector<Matrixf> resultMatrices;
				for (size_t j = 0; j < backMatrices.size(); j++)
					resultMatrices.push_back(backMatrices[j].MaxUpSample(get<0>(heigthWidth), get<1>(heigthWidth), downSampleIndices[i][j]));

				//cout << resultMatrices[0].GetString() << endl;

				backMatrices = resultMatrices;
			}

			for(size_t i = 0; i < backMatrices.size(); i++)
			{
				auto backMatrix = backMatrices[i];
				Matrixf networkMatrix(backMatrix.RowCount(), backMatrix.ColumnCount(), backNetworkResult.get() + i * backMatrix.RowCount() * backMatrix.ColumnCount());

				//cout << "Network matrix: " << endl << networkMatrix.GetString() << endl;
				//cout << "Back matrix: " << endl << backMatrix.GetString() << endl;
				auto difference = (backMatrix- networkMatrix).Norm2Square() / backMatrix.ElementCount();
				cout << "Back Difference: " << difference << endl;
				//cout << "Difference matrix: " << (backMatrix- networkMatrix).GetString() << endl;
				CHECK(difference < 1E-13f);
			}
		}

	}
}