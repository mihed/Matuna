/*
 * ConvNetConfig.h
 *
 *  Created on: May 2, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_MATUNA_CONVNET_CONVNETCONFIG_H_
#define MATUNA_MATUNA_CONVNET_CONVNETCONFIG_H_

#include "OutputLayerConfig.h"
#include "ForwardBackPropLayerConfig.h"
#include "LayerDescriptions.h"
#include <vector>
#include <memory>

using namespace std;

namespace Matuna
{
namespace MachineLearning
{

class ILayerConfigVisitor;

class ConvNetConfig: public ILayerConfig
{

private:
	vector<LayerDataDescription> inputDataDescriptions;

	vector<unique_ptr<ForwardBackPropLayerConfig>> forwardBackConfigs;
	unique_ptr<OutputLayerConfig> outputConfig;

	bool lowMemoryUsage;

public:
	ConvNetConfig(const vector<LayerDataDescription>& inputDataDescriptions, bool lowMemoryUsage = true);

	~ConvNetConfig();

	size_t LayerCount() const;

	bool HasOutputLayer() const;
	bool HasLowMemoryUsage() const;

	void SetLowMemoryUsage(bool value);

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
} /* namespace Matuna */

#endif /* MATUNA_MATUNA_CONVNET_CONVNETCONFIG_H_ */
