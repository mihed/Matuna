/*
 * CNNConfig.h
 *
 *  Created on: May 2, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNN_CNNCONFIG_H_
#define ATML_CNN_CNNCONFIG_H_

#include "OutputLayerConfig.h"
#include "ForwardBackPropLayerConfig.h"
#include "LayerDescriptions.h"
#include "ATMLPrecisionEnum.h"
#include <vector>
#include <memory>

using namespace std;

namespace ATML
{
namespace MachineLearning
{

class ILayerConfigVisitor;

class CNNConfig: public ILayerConfig
{

private:
	LayerMemoryDescription inputMemoryProposal;
	LayerDataDescription inputDataDescription;

	vector<unique_ptr<ForwardBackPropLayerConfig>> forwardBackConfigs;
	unique_ptr<OutputLayerConfig> outputConfig;

	ATMLPrecision precision;

public:
	CNNConfig(const LayerDataDescription& inputDataDescription,
			ATMLPrecision precision = ATMLSinglePrecision);
	CNNConfig(const LayerDataDescription& inputDataDescription,
			const LayerMemoryDescription& inputMemoryProposal,
			ATMLPrecision precision = ATMLSinglePrecision);
	~CNNConfig();

	size_t LayerCount() const
	{
		return forwardBackConfigs.size();
	}
	;

	ATMLPrecision Precision() const
	{
		return precision;
	}
	;

	bool HasOutputLayer() const
	{
		if (outputConfig.get())
			return true;
		else
			return false;
	}
	;

	void SetOutputConfig(unique_ptr<OutputLayerConfig> config);
	void RemoveOutputConfig();
	void AddToBack(unique_ptr<ForwardBackPropLayerConfig> config);
	void AddToFront(unique_ptr<ForwardBackPropLayerConfig> config);
	void InsertAt(unique_ptr<ForwardBackPropLayerConfig> config, size_t index);
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
