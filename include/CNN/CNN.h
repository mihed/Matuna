/*
 * CNN.h
 *
 *  Created on: May 2, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNN_CNN_H_
#define ATML_CNN_CNN_H_

#include "OpenCLHelper/OpenCLDevice.h"
#include "CNNConfig.h"
#include "OutputLayer.h"
#include "ForwardBackPropLayer.h"
#include <memory>
#include <vector>

using namespace std;
using namespace ATML::Helper;

namespace ATML {
namespace MachineLearning {

template<class T>
class CNN {
private:
	shared_ptr<OpenCLDevice> device;
	vector<unique_ptr<ForwardBackPropLayer>> forthBackLayers;
	unique_ptr<OutputLayer> outputLayer;
public:
	CNN(OpenCLDeviceInfo deviceInfo, CNNConfig config);
	~CNN();
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNN_CNN_H_ */
