/*
 * ForthBackPropLayerTest.cpp
 *
 *  Created on: May 3, 2015
 *      Author: Mikael
 */

#include "ForthBackPropLayerTest.h"
#include <chrono>
#include <random>

using namespace std;

ForthBackPropLayerTest::ForthBackPropLayerTest(
		const vector<LayerDataDescription>& inputLayerDescriptions,
		const ForwardBackPropLayerConfig* config) :
		ForwardBackPropLayer(inputLayerDescriptions, config)
{
	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
	default_random_engine generator(seed);
	uniform_int_distribution<int> unitGenerator(1, 100);
	uniform_int_distribution<int> paddingGenerator(1, 40);
	uniform_int_distribution<int> dimensionGenerator(1, 10000);

	//Now we just need to initialize the in/out memory proposal as well as setting
	//the output data descriptions.

	//The layers is entirely responsible for calculating how the output from this module looks like.
	//(This also defines the input of back prop module, which is the same of course - by definition!)

	LayerDataDescription inBackPropDataDescription;

	inBackPropDataDescription.Height = dimensionGenerator(generator);
	inBackPropDataDescription.Width = dimensionGenerator(generator);
	inBackPropDataDescription.Units = unitGenerator(generator);

	inBackPropDataDescriptions.push_back(inBackPropDataDescription);

	outForwardPropDataDescriptions = inBackPropDataDescriptions;

	//Setting the outForwardProp proposal

	LayerMemoryDescription outForwardPropMemoryProposal;

	outForwardPropMemoryProposal.HeightOffset = paddingGenerator(generator);
	outForwardPropMemoryProposal.Height =
			outForwardPropDataDescriptions[0].Height
					+ outForwardPropMemoryProposal.HeightOffset
					+ paddingGenerator(generator);

	outForwardPropMemoryProposal.WidthOffset = paddingGenerator(generator);
	outForwardPropMemoryProposal.Width = outForwardPropDataDescriptions[0].Width
			+ outForwardPropMemoryProposal.WidthOffset
			+ paddingGenerator(generator);

	outForwardPropMemoryProposal.UnitOffset = paddingGenerator(generator);
	outForwardPropMemoryProposal.Units = outForwardPropDataDescriptions[0].Units
			+ outForwardPropMemoryProposal.UnitOffset
			+ paddingGenerator(generator);

	outForwardPropMemoryProposals.push_back(outForwardPropMemoryProposal);

	//Setting the inBackProp proposal

	LayerMemoryDescription inBackPropMemoryProposal;

	inBackPropMemoryProposal.HeightOffset = paddingGenerator(generator);
	inBackPropMemoryProposal.Height = inBackPropDataDescription.Height
			+ inBackPropMemoryProposal.HeightOffset
			+ paddingGenerator(generator);

	inBackPropMemoryProposal.WidthOffset = paddingGenerator(generator);
	inBackPropMemoryProposal.Width = inBackPropDataDescription.Width
			+ inBackPropMemoryProposal.WidthOffset
			+ paddingGenerator(generator);

	inBackPropMemoryProposal.UnitOffset = paddingGenerator(generator);
	inBackPropMemoryProposal.Units = inBackPropDataDescription.Units
			+ inBackPropMemoryProposal.UnitOffset + paddingGenerator(generator);

	inBackPropMemoryProposals.push_back(inBackPropMemoryProposal);

	//Setting the inForwardProp proposal
	auto forwardInDataDescription = InForwardPropDataDescriptions();

	LayerMemoryDescription inForwardPropMemoryProposal;

	inForwardPropMemoryProposal.HeightOffset = paddingGenerator(generator);
	inForwardPropMemoryProposal.Height = forwardInDataDescription[0].Height
			+ inForwardPropMemoryProposal.HeightOffset
			+ paddingGenerator(generator);

	inForwardPropMemoryProposal.WidthOffset = paddingGenerator(generator);
	inForwardPropMemoryProposal.Width = forwardInDataDescription[0].Width
			+ inForwardPropMemoryProposal.WidthOffset
			+ paddingGenerator(generator);

	inForwardPropMemoryProposal.UnitOffset = paddingGenerator(generator);
	inForwardPropMemoryProposal.Units = forwardInDataDescription[0].Units
			+ inForwardPropMemoryProposal.UnitOffset
			+ paddingGenerator(generator);

	inForwardPropMemoryProposals.push_back(inForwardPropMemoryProposal);

	//Setting the outBackProp proposal

	LayerMemoryDescription outBackPropMemoryProposal;

	outBackPropMemoryProposal.HeightOffset = paddingGenerator(generator);
	outBackPropMemoryProposal.Height = forwardInDataDescription[0].Height
			+ outBackPropMemoryProposal.HeightOffset
			+ paddingGenerator(generator);

	outBackPropMemoryProposal.WidthOffset = paddingGenerator(generator);
	outBackPropMemoryProposal.Width = forwardInDataDescription[0].Width
			+ outBackPropMemoryProposal.WidthOffset
			+ paddingGenerator(generator);

	outBackPropMemoryProposal.UnitOffset = paddingGenerator(generator);
	outBackPropMemoryProposal.Units = forwardInDataDescription[0].Units
			+ outBackPropMemoryProposal.UnitOffset
			+ paddingGenerator(generator);

	outBackPropMemoryProposals.push_back(outBackPropMemoryProposal);

}

void ForthBackPropLayerTest::InterlockFinalized()
{

}

ForthBackPropLayerTest::~ForthBackPropLayerTest()
{

}

