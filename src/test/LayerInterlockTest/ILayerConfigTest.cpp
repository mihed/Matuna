/*
 * ILayerConfigTest.cpp
 *
 *  Created on: May 3, 2015
 *      Author: Mikael
 */

#include "ILayerConfigTest.h"
#include "CNN/ILayerConfigVisitor.h"
#include "FactoryVisitorTest.h"

ILayerConfigTest::ILayerConfigTest()
{
	// TODO Auto-generated constructor stub
}

ILayerConfigTest::~ILayerConfigTest()
{
	// TODO Auto-generated destructor stub
}

void ILayerConfigTest::Accept(ILayerConfigVisitor* visitor)
{
	this->Accept(dynamic_cast<FactoryVisitorTest*>(visitor));
}
