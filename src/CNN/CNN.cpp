/*
 * CNN.cpp
 *
 *  Created on: May 2, 2015
 *      Author: Mikael
 */

#include "CNN.h"

namespace ATML
{
namespace MachineLearning
{

CNN::CNN(const CNNConfig& config)
{

}

CNN::~CNN()
{

}

vector<LayerDataDescription> CNN::InputDataDescriptions() const
{
	return inputDataDescriptions;
}

vector<LayerMemoryDescription> CNN::InputMemoryDescriptions() const
{
	return inputMemoryDescriptions;
}

} /* namespace MachineLearning */
} /* namespace ATML */
