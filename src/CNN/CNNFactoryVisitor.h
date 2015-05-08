/*
 * CNNFactoryVisitor.h
 *
 *  Created on: May 5, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNN_CNNFACTORYVISITOR_H_
#define ATML_CNN_CNNFACTORYVISITOR_H_

#include "ILayerConfigVisitor.h"
#include "ForwardBackPropLayer.h"
#include "OutputLayer.h"
#include <vector>
#include <memory>

using namespace std;

namespace ATML
{
namespace MachineLearning
{

class CNNFactoryVisitor: public ILayerConfigVisitor
{
protected:
	vector<unique_ptr<ForwardBackPropLayer>> layers;
	unique_ptr<OutputLayer> outputLayer;

	vector<LayerDataDescription> inputDataDescriptions;
	vector<LayerMemoryDescription> forwardInputProposals;
	vector<LayerMemoryDescription> backOutputProposals;

	void InterlockLayer(ForwardBackPropLayer* layer);
	void InterlockLayer(BackPropLayer* layer);

public:
	CNNFactoryVisitor();
	virtual ~CNNFactoryVisitor();

	vector<unique_ptr<ForwardBackPropLayer>> GetLayers();
	unique_ptr<OutputLayer> GetOutputLayer();
};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNN_CNNFACTORYVISITOR_H_ */
