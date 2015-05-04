/*
 * CNNConfig.h
 *
 *  Created on: May 2, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNN_CNNCONFIG_H_
#define ATML_CNN_CNNCONFIG_H_

#include "ILayerConfig.h"
#include "LayerDescriptions.h"
#include <vector>
#include <memory>

using namespace std;

namespace ATML
{
namespace MachineLearning
{

class ILayerConfigVisitor;

class CNNConfig: ILayerConfig
{
private:
	LayerMemoryDescription inputMemoryProposal;
	LayerDataDescription inputDataDescription;

	vector<unique_ptr<ILayerConfig>> configs;

public:
	CNNConfig(const LayerDataDescription& inputDataDescription);
	CNNConfig(const LayerDataDescription& inputDataDescription,
			const LayerMemoryDescription& inputMemoryProposal);
	~CNNConfig();

	size_t LayerCount()
	{
		return configs.size();
	}
	;

	void AddToBack(unique_ptr<ILayerConfig> config);
	void AddToFront(unique_ptr<ILayerConfig> config);
	void InsertAt(unique_ptr<ILayerConfig> config, size_t index);
	void RemoveAt(size_t index);

	virtual void Accept(ILayerConfigVisitor* visitor) override;

	LayerMemoryDescription InputMemoryProposal() const
	{
		return inputMemoryProposal;
	}
	;

	LayerDataDescription InputDataDescription() const
	{
		return inputDataDescription;
	}
	;
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNN_CNNCONFIG_H_ */
