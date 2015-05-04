/*
 * OutputLayerConfig.cpp
 *
 *  Created on: May 3, 2015
 *      Author: Mikael
 */

#include "OutputLayerConfigTest.h"
#include "FactoryVisitorTest.h"

OutputLayerConfigTest::OutputLayerConfigTest()
{

}

OutputLayerConfigTest::~OutputLayerConfigTest()
{

}

void OutputLayerConfigTest::Accept(FactoryVisitorTest* visitor)
{
	visitor->Visit(this);
}

