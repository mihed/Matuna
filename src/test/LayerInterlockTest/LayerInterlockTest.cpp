/*
 * LayerInterlockTest.cpp
 *
 *  Created on: May 3, 2015
 *      Author: Mikael
 */
#define CATCH_CONFIG_MAIN
#include "catch/catch.hpp"

#include "CNN/CNNConfig.h"
#include "CNN/ILayerConfig.h"
#include "CNN/ILayerConfigVisitor.h"
#include "CNN/InterlockHelper.h"

#include "ForthBackPropLayerConfigTest.h"
#include "ForthBackPropLayerTest.h"
#include "OutputLayerConfigTest.h"
#include "OutputLayerTest.h"
#include "FactoryVisitorTest.h"

#include <memory>
#include <chrono>
#include <random>

using namespace std;
using namespace ATML::MachineLearning;

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

SCENARIO("Creating compatible memory with the interlock helper", "[InterlockHelper]")
{
	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
	default_random_engine generator(seed);
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
	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
	default_random_engine generator(seed);
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

		CNNConfig cnnConfig(inputDescription);

		for (int i = 0; i < count; i++)
		{
			auto fbConfig = new ForthBackPropLayerConfigTest();
			unique_ptr<ILayerConfig> config1(fbConfig);
			cnnConfig.AddToBack(move(config1));
		}

		auto outputConfig = new OutputLayerConfigTest();
		unique_ptr<ILayerConfig> oConfig(outputConfig);
		cnnConfig.AddToBack(move(oConfig));

		INFO("Visiting the factory");
		FactoryVisitorTest factory;
		cnnConfig.Accept(&factory);

		WHEN("Fetching all the layers from the factory")
		{
			auto layers = factory.GetLayers();
			THEN("The pairwise memory and unit descriptions must match")
			{
				for (int i = 0; i < count - 1; i++)
				{
					auto layer1 = dynamic_cast<ForthBackPropLayerTest*>(layers[i].get());
					auto layer2 = dynamic_cast<ForthBackPropLayerTest*>(layers[i + 1].get());
					INFO("Making sure that the pointer is valid");
					CHECK(layer1);
					CHECK(layer2);

					INFO("Making sure that the layers are interlocked");
					CHECK(layer1->Interlocked());
					CHECK(layer2->Interlocked());

					INFO("Making sure that we get an exception if we try to interlock");
					CHECK_THROWS(layer1->InterlockBackPropInput(LayerMemoryDescription()));
					CHECK_THROWS(layer1->InterlockBackPropOutput(LayerMemoryDescription()));
					CHECK_THROWS(layer1->InterlockForwardPropInput(LayerMemoryDescription()));
					CHECK_THROWS(layer1->InterlockForwardPropOutput(LayerMemoryDescription()));
					CHECK_THROWS(layer2->InterlockBackPropInput(LayerMemoryDescription()));
					CHECK_THROWS(layer2->InterlockBackPropOutput(LayerMemoryDescription()));
					CHECK_THROWS(layer2->InterlockForwardPropInput(LayerMemoryDescription()));
					CHECK_THROWS(layer2->InterlockForwardPropOutput(LayerMemoryDescription()));

					CheckMemoryDescription(layer1->InBackPropMemoryDescription(), layer2->OutBackPropMemoryDescription());
					CheckMemoryDescription(layer1->OutForwardPropMemoryDescription(), layer2->InForwardPropMemoryDescription());
					CheckUnitDescription(layer1->InBackPropDataDescription(), layer2->OutBackPropDataDescription());
					CheckUnitDescription(layer1->OutForwardPropDataDescription(), layer2->InForwardPropDataDescription());

					bool result = InterlockHelper::IsCompatible(layer1->InBackPropMemoryDescription(), layer1->InBackPropMemoryProposal());
					CHECK(result);
					result = InterlockHelper::IsCompatible(layer1->OutBackPropMemoryDescription(), layer1->OutBackPropMemoryProposal());
					CHECK(result);
					result = InterlockHelper::IsCompatible(layer1->InForwardPropMemoryDescription(), layer1->InForwardPropMemoryProposal());
					CHECK(result);
					result = InterlockHelper::IsCompatible(layer1->OutForwardPropMemoryDescription(), layer1->OutForwardPropMemoryProposal());
					CHECK(result);
					result = InterlockHelper::IsCompatible(layer2->InBackPropMemoryDescription(), layer2->InBackPropMemoryProposal());
					CHECK(result);
					result = InterlockHelper::IsCompatible(layer2->OutBackPropMemoryDescription(), layer2->OutBackPropMemoryProposal());
					CHECK(result);
					result = InterlockHelper::IsCompatible(layer2->InForwardPropMemoryDescription(), layer2->InForwardPropMemoryProposal());
					CHECK(result);
					result = InterlockHelper::IsCompatible(layer2->OutForwardPropMemoryDescription(), layer2->OutForwardPropMemoryProposal());
					CHECK(result);
				}

				INFO("Checking the output layer");
				auto lastLayer = dynamic_cast<ForthBackPropLayerTest*>(layers[layers.size() - 2].get());
				auto outputLayer = dynamic_cast<OutputLayerTest*>(layers[layers.size() - 1].get());

				CHECK(lastLayer);
				CHECK(outputLayer);
				CHECK(outputLayer->Interlocked());
				CHECK_THROWS(outputLayer->InterlockBackPropInput(LayerMemoryDescription()));
				CHECK_THROWS(outputLayer->InterlockBackPropOutput(LayerMemoryDescription()));
				CHECK_THROWS(outputLayer->InterlockForwardPropInput(LayerMemoryDescription()));
				CheckMemoryDescription(lastLayer->InBackPropMemoryDescription(), outputLayer->OutBackPropMemoryDescription());
				CheckMemoryDescription(lastLayer->OutForwardPropMemoryDescription(), outputLayer->InForwardPropMemoryDescription());
				CheckUnitDescription(lastLayer->InBackPropDataDescription(), outputLayer->OutBackPropDataDescription());
				CheckUnitDescription(lastLayer->OutForwardPropDataDescription(), outputLayer->InForwardPropDataDescription());
			}
		}

	}
}

