/*
 * FactoryVisitorTest.h
 *
 *  Created on: May 3, 2015
 *      Author: Mikael
 */

#ifndef ATML_TEST_LAYERINTERLOCKTEST_FACTORYVISITORTEST_H_
#define ATML_TEST_LAYERINTERLOCKTEST_FACTORYVISITORTEST_H_

#include "CNN/ILayerConfigVisitor.h"
#include "CNN/LayerDescriptions.h"
#include "CNN/ForwardBackPropLayer.h"
#include "CNN/BackPropLayer.h"
#include <vector>
#include <memory>

class OutputLayerConfigTest;
class ForthBackPropLayerConfigTest;

using namespace std;
using namespace ATML::MachineLearning;

class FactoryVisitorTest: public ILayerConfigVisitor
{
private:
	vector<unique_ptr<BackPropLayer>> layers;

	LayerDataDescription inputDataDescription;
	LayerMemoryDescription forwardInputProposal;
	LayerMemoryDescription backOutputProposal;

	void InterlockLayer(ForwardBackPropLayer* layer);
	void InterlockLayer(BackPropLayer* layer);
public:
	FactoryVisitorTest();
	~FactoryVisitorTest();

	virtual void Visit(const CNNConfig* const cnnConfig) override;
	void Visit(const OutputLayerConfigTest* const);
	void Visit(const ForthBackPropLayerConfigTest* const);

	vector<unique_ptr<BackPropLayer>> GetLayers();
};

#endif /* ATML_TEST_LAYERINTERLOCKTEST_FACTORYVISITORTEST_H_ */
