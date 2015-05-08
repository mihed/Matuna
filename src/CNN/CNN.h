/*
 * CNN.h
 *
 *  Created on: May 2, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNN_CNN_H_
#define ATML_CNN_CNN_H_

#include "CNNConfig.h"
#include "LayerDescriptions.h"

#include <vector>

using namespace std;

namespace ATML
{
namespace MachineLearning
{

class CNN
{
private:
	vector<LayerDataDescription> inputDataDescriptions;
	vector<LayerMemoryDescription> inputMemoryDescriptions;
	vector<LayerMemoryDescription> inputMemoryProposals;
	bool inputInterlocked;
public:
	CNN(const CNNConfig& config);
	virtual ~CNN();

	void InterlockForwardPropInput(
			const vector<LayerMemoryDescription>& inputDescriptions);

	bool Interlocked() const;

	vector<LayerDataDescription> InputDataDescriptions() const;
	vector<LayerMemoryDescription> InputMemoryDescriptions() const;
	vector<LayerMemoryDescription> InputMemoryProposals() const;

};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNN_CNN_H_ */
