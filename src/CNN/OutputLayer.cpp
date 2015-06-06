/*
 * OutputLayer.cpp
 *
 *  Created on: May 2, 2015
 *      Author: Mikael
 */

#include "OutputLayer.h"
#include <stdexcept>

using namespace std;

namespace Matuna
{
namespace MachineLearning
{

OutputLayer::OutputLayer(
		const vector<LayerDataDescription>& inputLayerDescriptions,
		MatunaActivationFunction backPropActivation,
		const OutputLayerConfig* outputLayerConfig) :
		BackPropLayer(inputLayerDescriptions, backPropActivation)
{

}

OutputLayer::~OutputLayer()
{

}
}
/* namespace MachineLearning */
} /* namespace Matuna */
