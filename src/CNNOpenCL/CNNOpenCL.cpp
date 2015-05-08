/*
 * CNNOpenCL.cpp
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#include "CNNOpenCL.h"

namespace ATML
{
namespace MachineLearning
{

CNNOpenCL::CNNOpenCL(unique_ptr<OpenCLContext> context, const CNNConfig& config) :
		CNN(config)
{
	this->context = move(context);
}

CNNOpenCL::~CNNOpenCL()
{

}

} /* namespace MachineLearning */
} /* namespace ATML */
