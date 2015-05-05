/*
 * OutputLayerTest.cpp
 *
 *  Created on: May 3, 2015
 *      Author: Mikael
 */

#include "OutputLayerTest.h"
#include <chrono>
#include <random>

using namespace std;

OutputLayerTest::OutputLayerTest(
		const LayerDataDescription& inputLayerDescription,
		const OutputLayerConfig* config) :
		OutputLayer(inputLayerDescription, config)
{
	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
	default_random_engine generator(seed);
	uniform_int_distribution<int> unitGenerator(1, 100);
	uniform_int_distribution<int> paddingGenerator(1, 40);
	uniform_int_distribution<int> dimensionGenerator(1, 10000);

	//The layers is entirely responsible for calculating how the output from this module looks like.
	//(This also defines the input of back prop module, which is the same of course - by definition!)
	inBackPropDataDescription.Height = dimensionGenerator(generator);
	inBackPropDataDescription.Width = dimensionGenerator(generator);
	inBackPropDataDescription.Units = unitGenerator(generator);

	//Setting the inBackProp proposal. This is actually the target memory!
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

	//Setting the inForwardProp proposal
	auto forwardInDataDescription = InForwardPropDataDescription();
	inForwardPropMemoryProposal.HeightOffset = paddingGenerator(generator);
	inForwardPropMemoryProposal.Height = forwardInDataDescription.Height
			+ inForwardPropMemoryProposal.HeightOffset
			+ paddingGenerator(generator);

	inForwardPropMemoryProposal.WidthOffset = paddingGenerator(generator);
	inForwardPropMemoryProposal.Width = forwardInDataDescription.Width
			+ inForwardPropMemoryProposal.WidthOffset
			+ paddingGenerator(generator);

	inForwardPropMemoryProposal.UnitOffset = paddingGenerator(generator);
	inForwardPropMemoryProposal.Units = forwardInDataDescription.Units
			+ inForwardPropMemoryProposal.UnitOffset
			+ paddingGenerator(generator);

	//Setting the outBackProp proposal
	outBackPropMemoryProposal.HeightOffset = paddingGenerator(generator);
	outBackPropMemoryProposal.Height = forwardInDataDescription.Height
			+ outBackPropMemoryProposal.HeightOffset
			+ paddingGenerator(generator);

	outBackPropMemoryProposal.WidthOffset = paddingGenerator(generator);
	outBackPropMemoryProposal.Width = forwardInDataDescription.Width
			+ outBackPropMemoryProposal.WidthOffset
			+ paddingGenerator(generator);

	outBackPropMemoryProposal.UnitOffset = paddingGenerator(generator);
	outBackPropMemoryProposal.Units = forwardInDataDescription.Units
			+ outBackPropMemoryProposal.UnitOffset
			+ paddingGenerator(generator);

}

OutputLayerTest::~OutputLayerTest()
{

}

