/*
 * ForthBackPropLayerConfigTest.cpp
 *
 *  Created on: May 3, 2015
 *      Author: Mikael
 */

#include "ForthBackPropLayerConfigTest.h"
#include "FactoryVisitorTest.h"

ForthBackPropLayerConfigTest::ForthBackPropLayerConfigTest()
{


}

ForthBackPropLayerConfigTest::~ForthBackPropLayerConfigTest()
{

}

void ForthBackPropLayerConfigTest::Accept(FactoryVisitorTest* visitor)
{
	visitor->Visit(this);
}

