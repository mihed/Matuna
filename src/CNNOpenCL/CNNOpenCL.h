/*
 * CNNOpenCL.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNNOPENCL_CNNOPENCL_H_
#define ATML_CNNOPENCL_CNNOPENCL_H_

#include "CNN/CNNConfig.h"
#include "CNN/CNN.h"

namespace ATML
{
namespace MachineLearning
{

class CNNOpenCL: public CNN
{
public:
	CNNOpenCL(const CNNConfig& config);
	~CNNOpenCL();
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNNOPENCL_CNNOPENCL_H_ */
