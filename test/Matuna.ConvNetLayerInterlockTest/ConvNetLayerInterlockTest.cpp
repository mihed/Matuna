/*
* LayerInterlockTest.cpp
*
*  Created on: May 3, 2015
*      Author: Mikael
*/
#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"

#include "Matuna.ConvNet/ConvNetConfig.h"
#include "Matuna.ConvNet/ILayerConfig.h"
#include "Matuna.ConvNet/ILayerConfigVisitor.h"
#include "Matuna.ConvNet/InterlockHelper.h"
#include "Matuna.ConvNet/ForwardBackPropLayerConfig.h"
#include "Matuna.ConvNet/OutputLayerConfig.h"
#include "Matuna.ConvNet/PerceptronLayerConfig.h"
#include "Matuna.ConvNet/ConvolutionLayerConfig.h"
#include "Matuna.ConvNet/StandardOutputLayerConfig.h"
#include "Matuna.ConvNet/ConvNet.h"

#include "ForthBackPropLayerTest.h"
#include "OutputLayerTest.h"
#include "ConvNetFactoryVisitorTest.h"

#include <memory>
#include <chrono>
#include <random>

using namespace std;
using namespace Matuna::MachineLearning;

void CheckUnitDescription(const LayerDataDescription& left,
						  const LayerDataDescription& right)
{
	CHECK(left.Height == right.Height);
	CHECK(left.Width == right.Width);
	CHECK(left.Units == right.Units);

	CHECK(right.Height >= 0);
	CHECK(right.Width >= 0);
	CHECK(right.Units >= 0);

	CHECK(left.Height >= 0);
	CHECK(left.Width >= 0);
	CHECK(left.Units >= 0);
}

void CheckUnitDescription(const vector<LayerDataDescription>& left,
						  const vector<LayerDataDescription>& right)
{
	auto count = left.size();
	CHECK(count == right.size());
	for (size_t i = 0; i < count; i++)
		CheckUnitDescription(left[i], right[i]);
}

void CheckMemoryDescription(const LayerMemoryDescription& left,
							const LayerMemoryDescription& right)
{
	CHECK(left.Height == right.Height);
	CHECK(left.Width == right.Width);
	CHECK(left.Units == right.Units);
	CHECK(left.HeightOffset == right.HeightOffset);
	CHECK(left.WidthOffset == right.WidthOffset);
	CHECK(left.UnitOffset == right.UnitOffset);

	CHECK(left.Height >= 0);
	CHECK(left.Width >= 0);
	CHECK(left.Units >= 0);
	CHECK(left.HeightOffset >= 0);
	CHECK(left.WidthOffset >= 0);
	CHECK(left.UnitOffset >= 0);

	CHECK(right.Height >= 0);
	CHECK(right.Width >= 0);
	CHECK(right.Units >= 0);
	CHECK(right.HeightOffset >= 0);
	CHECK(right.WidthOffset >= 0);
	CHECK(right.UnitOffset >= 0);

}

void CheckMemoryDescription(const vector<LayerMemoryDescription>& left,
							const vector<LayerMemoryDescription>& right)
{
	auto count = left.size();
	CHECK(count == right.size());
	for (size_t i = 0; i < count; i++)
		CheckMemoryDescription(left[i], right[i]);
}

SCENARIO("Creating compatible memory with the interlock helper", "[InterlockHelper]")
{
	random_device tempDevice;
	mt19937 generator(tempDevice());
	uniform_int_distribution<int> dimensionGenerator(1, 10000);

	int tests = 10000;
	for (int i = 0; i < tests; i++)
	{
		LayerMemoryDescription desc1;
		desc1.Height = dimensionGenerator(generator);
		desc1.Width = dimensionGenerator(generator);
		desc1.Units = dimensionGenerator(generator);
		desc1.HeightOffset = dimensionGenerator(generator);
		desc1.WidthOffset = dimensionGenerator(generator);
		desc1.UnitOffset = dimensionGenerator(generator);

		LayerMemoryDescription desc2;
		desc2.Height = dimensionGenerator(generator);
		desc2.Width = dimensionGenerator(generator);
		desc2.Units = dimensionGenerator(generator);
		desc2.HeightOffset = dimensionGenerator(generator);
		desc2.WidthOffset = dimensionGenerator(generator);
		desc2.UnitOffset = dimensionGenerator(generator);

		if (desc1.HeightOffset > desc1.Height)
			CHECK_THROWS(InterlockHelper::CalculateCompatibleMemory(desc1, desc2));
		else if (desc1.WidthOffset > desc1.Width)
			CHECK_THROWS(InterlockHelper::CalculateCompatibleMemory(desc1, desc2));
		else if (desc1.UnitOffset > desc1.Units)
			CHECK_THROWS(InterlockHelper::CalculateCompatibleMemory(desc1, desc2));
		else if (desc2.HeightOffset > desc2.Height)
			CHECK_THROWS(InterlockHelper::CalculateCompatibleMemory(desc1, desc2));
		else if (desc2.WidthOffset > desc2.Width)
			CHECK_THROWS(InterlockHelper::CalculateCompatibleMemory(desc1, desc2));
		else if (desc2.UnitOffset > desc2.Units)
			CHECK_THROWS(InterlockHelper::CalculateCompatibleMemory(desc1, desc2));
		else
		{
			auto result = InterlockHelper::CalculateCompatibleMemory(desc1, desc2);
			CHECK(result.Height >= desc1.Height);
			CHECK(result.Width >= desc1.Width);
			CHECK(result.Units >= desc1.Units);
			CHECK(result.UnitOffset >= desc1.UnitOffset);
			CHECK(result.WidthOffset >= desc1.WidthOffset);
			CHECK(result.HeightOffset >= desc1.HeightOffset);

			int test1, test2;
			test1 = desc1.Height - desc1.HeightOffset;
			test2 = result.Height - result.HeightOffset;
			CHECK(test2 >= test1);
			test1 = desc1.Width - desc1.WidthOffset;
			test2 = result.Width - result.WidthOffset;
			CHECK(test2 >= test1);
			test1 = desc1.Units - desc1.UnitOffset;
			test2 = result.Units - result.UnitOffset;
			CHECK(test2 >= test1);

			test1 = desc2.Height - desc2.HeightOffset;
			test2 = result.Height - result.HeightOffset;
			CHECK(test2 >= test1);
			test1 = desc2.Width - desc2.WidthOffset;
			test2 = result.Width - result.WidthOffset;
			CHECK(test2 >= test1);
			test1 = desc2.Units - desc2.UnitOffset;
			test2 = result.Units - result.UnitOffset;
			CHECK(test2 >= test1);

			CHECK(result.Height >= desc2.Height);
			CHECK(result.Width >= desc2.Width);
			CHECK(result.Units >= desc2.Units);
			CHECK(result.UnitOffset >= desc2.UnitOffset);
			CHECK(result.WidthOffset >= desc2.WidthOffset);
			CHECK(result.HeightOffset >= desc2.HeightOffset);
		}
	}
}

SCENARIO("Creating a network from configurations.", "[InterlockHelper][ILayerConfig][ILayerConfigVisitor]")
{
	random_device tempDevice;
	mt19937 generator(tempDevice());
	uniform_int_distribution<int> unitGenerator(1, 100);
	uniform_int_distribution<int> dimensionGenerator(1, 10000);
	uniform_int_distribution<int> numConfigsGenerator(1, 100);

	int count = numConfigsGenerator(generator);

	GIVEN("Different configurations")
	{

		INFO("Creating random configurations");
		LayerDataDescription inputDescription;
		inputDescription.Height = dimensionGenerator(generator);
		inputDescription.Width = dimensionGenerator(generator);
		inputDescription.Units = unitGenerator(generator);
		vector<LayerDataDescription> inputDescriptions;
		inputDescriptions.push_back(inputDescription);
		ConvNetConfig cnnConfig(inputDescriptions);

		for (int i = 0; i < count; i++)
		{
			unique_ptr<ForwardBackPropLayerConfig> config1(new PerceptronLayerConfig(10));
			cnnConfig.AddToBack(move(config1));
			unique_ptr<ForwardBackPropLayerConfig> config2(new ConvolutionLayerConfig(1, 1, 1));
			cnnConfig.AddToBack(move(config2));
		}

		unique_ptr<OutputLayerConfig> oConfig(new StandardOutputLayerConfig());
		cnnConfig.SetOutputConfig(move(oConfig));

		INFO("Visiting the factory");
		ConvNet networkDummy(cnnConfig);
		ConvNetFactoryVisitorTest factory(&networkDummy);
		cnnConfig.Accept(&factory);

		WHEN("Fetching all the layers from the factory")
		{
			auto layers = factory.GetLayers();
			THEN("The pairwise memory and unit descriptions must match")
			{
				for (int i = 0; i < count - 1; i++)
				{
					auto layer1 = layers[i].get();
					auto layer2 = layers[i + 1].get();
					INFO("Making sure that the pointer is valid");
					CHECK(layer1);
					CHECK(layer2);

					INFO("Making sure that the layers are interlocked");
					CHECK(layer1->Interlocked());
					CHECK(layer2->Interlocked());

					INFO("Making sure that we get an exception if we try to interlock");
					vector<LayerMemoryDescription> dummyTest;
					dummyTest.push_back(LayerMemoryDescription());
					CHECK_THROWS(layer1->InterlockBackPropInput(dummyTest));
					CHECK_THROWS(layer1->InterlockBackPropOutput(dummyTest));
					CHECK_THROWS(layer1->InterlockForwardPropInput(dummyTest));
					CHECK_THROWS(layer1->InterlockForwardPropOutput(dummyTest));
					CHECK_THROWS(layer2->InterlockBackPropInput(dummyTest));
					CHECK_THROWS(layer2->InterlockBackPropOutput(dummyTest));
					CHECK_THROWS(layer2->InterlockForwardPropInput(dummyTest));
					CHECK_THROWS(layer2->InterlockForwardPropOutput(dummyTest));

					CheckMemoryDescription(layer1->InBackPropMemoryDescriptions(), layer2->OutBackPropMemoryDescriptions());
					CheckMemoryDescription(layer1->OutForwardPropMemoryDescriptions(), layer2->InForwardPropMemoryDescriptions());
					CheckUnitDescription(layer1->InBackPropDataDescriptions(), layer2->OutBackPropDataDescriptions());
					CheckUnitDescription(layer1->OutForwardPropDataDescriptions(), layer2->InForwardPropDataDescriptions());

					bool result = InterlockHelper::IsCompatible(layer1->InBackPropMemoryDescriptions(), layer1->InBackPropMemoryProposals());
					CHECK(result);
					result = InterlockHelper::IsCompatible(layer1->OutBackPropMemoryDescriptions(), layer1->OutBackPropMemoryProposals());
					CHECK(result);
					result = InterlockHelper::IsCompatible(layer1->InForwardPropMemoryDescriptions(), layer1->InForwardPropMemoryProposals());
					CHECK(result);
					result = InterlockHelper::IsCompatible(layer1->OutForwardPropMemoryDescriptions(), layer1->OutForwardPropMemoryProposals());
					CHECK(result);
					result = InterlockHelper::IsCompatible(layer2->InBackPropMemoryDescriptions(), layer2->InBackPropMemoryProposals());
					CHECK(result);
					result = InterlockHelper::IsCompatible(layer2->OutBackPropMemoryDescriptions(), layer2->OutBackPropMemoryProposals());
					CHECK(result);
					result = InterlockHelper::IsCompatible(layer2->InForwardPropMemoryDescriptions(), layer2->InForwardPropMemoryProposals());
					CHECK(result);
					result = InterlockHelper::IsCompatible(layer2->OutForwardPropMemoryDescriptions(), layer2->OutForwardPropMemoryProposals());
					CHECK(result);
				}

				INFO("Checking the output layer");
				auto lastLayer = layers[layers.size() - 1].get();
				auto outputLayerPointer = factory.GetOutputLayer();
				auto outputLayer = outputLayerPointer.get();

				CHECK(lastLayer);
				CHECK(outputLayer);
				CHECK(outputLayer->Interlocked());
				vector<LayerMemoryDescription> anotherDummy;
				anotherDummy.push_back(LayerMemoryDescription());
				CHECK_THROWS(outputLayer->InterlockBackPropInput(anotherDummy));
				CHECK_THROWS(outputLayer->InterlockBackPropOutput(anotherDummy));
				CHECK_THROWS(outputLayer->InterlockForwardPropInput(anotherDummy));
				CheckMemoryDescription(lastLayer->InBackPropMemoryDescriptions(), outputLayer->OutBackPropMemoryDescriptions());
				CheckMemoryDescription(lastLayer->OutForwardPropMemoryDescriptions(), outputLayer->InForwardPropMemoryDescriptions());
				CheckUnitDescription(lastLayer->InBackPropDataDescriptions(), outputLayer->OutBackPropDataDescriptions());
				CheckUnitDescription(lastLayer->OutForwardPropDataDescriptions(), outputLayer->InForwardPropDataDescriptions());
			}
		}

	}
}

