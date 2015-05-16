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
	vector<LayerDataDescription> inputDataDescriptions;

	vector<unique_ptr<ForwardBackPropLayerConfig>> forwardBackConfigs;
	unique_ptr<OutputLayerConfig> outputConfig;

public:
	CNNConfig(const vector<LayerDataDescription>& inputDataDescriptions);

	~CNNConfig();

	size_t LayerCount() const;

	bool HasOutputLayer() const;
	void SetOutputConfig(unique_ptr<OutputLayerConfig> config);
	void RemoveOutputConfig();
	void AddToBack(unique_ptr<ForwardBackPropLayerConfig> config);
	void AddToFront(unique_ptr<ForwardBackPropLayerConfig> config);
	void InsertAt(unique_ptr<ForwardBackPropLayerConfig> config, size_t index);
	void RemoveAt(size_t index);

	virtual void Accept(ILayerConfigVisitor* visitor) override;

	vector<LayerDataDescription> InputDataDescription() const;
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNN_CNNCONFIG_H_ */
