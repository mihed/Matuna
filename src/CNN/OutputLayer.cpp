/*
 * OutputLayer.cpp
 *
 *  Created on: May 2, 2015
 *      Author: Mikael
 */

#include "OutputLayer.h"
#include <stdexcept>

using namespace std;

namespace ATML
{
namespace MachineLearning
{

OutputLayer::OutputLayer(const LayerDataDescription& inputLayerDescription) :
		BackPropLayer(inputLayerDescription)
{

}

OutputLayer::~OutputLayer()
{

}
}
/* namespace MachineLearning */
} /* namespace ATML */
