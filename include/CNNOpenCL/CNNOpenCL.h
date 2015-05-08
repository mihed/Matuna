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
#include "OpenCLHelper/OpenCLContext.h"

#include <memory>

using namespace std;
using namespace ATML::Helper;

namespace ATML
{
namespace MachineLearning
{

class CNNOpenCL: public CNN
{
private:
	shared_ptr<OpenCLContext> context;
public:
	CNNOpenCL(unique_ptr<OpenCLContext> context, const CNNConfig& config);
	~CNNOpenCL();
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNNOPENCL_CNNOPENCL_H_ */
