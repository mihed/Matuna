/*
 * ConvNetFactoryVisitor.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_CONVNET_CONVNETFACTORYVISITOR_H_
#define MATUNA_CONVNET_CONVNETFACTORYVISITOR_H_

#include "ILayerConfigVisitor.h"
#include "ForwardBackPropLayer.h"
#include "MatunaActivationFunctionEnum.h"
#include "OutputLayer.h"
#include <vector>
#include <memory>

using namespace std;

namespace Matuna
{
namespace MachineLearning
{

class ConvNet;

class ConvNetFactoryVisitor: public ILayerConfigVisitor
{
private:
	bool outputIsCalled;


	void InterlockLayer(ForwardBackPropLayer* layer);
	void InterlockLayer(BackPropLayer* layer);

protected:
	ConvNet* network;

	vector<unique_ptr<ForwardBackPropLayer>> layers;
	unique_ptr<OutputLayer> outputLayer;

	vector<LayerDataDescription> inputDataDescriptions;
	vector<LayerMemoryDescription> forwardInputProposals;
	vector<LayerMemoryDescription> backOutputProposals;
	MatunaActivationFunction backPropActivation;

	void InterlockAndAddLayer(const PerceptronLayerConfig* const config, unique_ptr<ForwardBackPropLayer> layer);
	void InterlockAndAddLayer(const ConvolutionLayerConfig* const config, unique_ptr<ForwardBackPropLayer> layer);
	void InterlockAndAddLayer(const StandardOutputLayerConfig* const config, unique_ptr<OutputLayer> layer);
	void InterlockAndAddLayer(const VanillaSamplingLayerConfig* const config, unique_ptr<ForwardBackPropLayer> layer);
	void InterlockAndAddLayer(const MaxPoolingLayerConfig* const config, unique_ptr<ForwardBackPropLayer> layer);
	void InitializeInterlock(const ConvNetConfig* const config);

public:
	ConvNetFactoryVisitor(ConvNet* network);
	virtual ~ConvNetFactoryVisitor();

	vector<unique_ptr<ForwardBackPropLayer>> GetLayers();
	unique_ptr<OutputLayer> GetOutputLayer();
};

} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_CONVNET_CONVNETFACTORYVISITOR_H_ */
