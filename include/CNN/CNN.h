/*
 * CNN.h
 *
 *  Created on: May 2, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNN_CNN_H_
#define ATML_CNN_CNN_H_

#include "CNNConfig.h"

namespace ATML {
namespace MachineLearning {

template<class T>
class CNN {
public:
	CNN(const CNNConfig& config);
	~CNN();
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNN_CNN_H_ */
