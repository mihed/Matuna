/*
 * InterlockHelper.h
 *
 *  Created on: May 3, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNN_INTERLOCKHELPER_H_
#define ATML_CNN_INTERLOCKHELPER_H_

namespace ATML
{
namespace MachineLearning
{

//Forward declarations
class ForwardBackPropLayer;
class OutputLayer;
class CNNConfig;
class LayerMemoryDescription;
class LayerDataDescription;

class InterlockHelper
{
public:
	static LayerMemoryDescription CalculateCompatibleMemory(
			const LayerMemoryDescription& right,
			const LayerMemoryDescription& left);

	static bool IsCompatible(const LayerMemoryDescription& right,
			const LayerMemoryDescription& left);
	static bool IsCompatible(const LayerDataDescription& data,
			const LayerMemoryDescription& memory);

	static void Interlock(ForwardBackPropLayer* left, OutputLayer* right);
	static void Interlock(ForwardBackPropLayer* left,
			ForwardBackPropLayer* right);
	static void Interlock(const CNNConfig& inputConfig,
			ForwardBackPropLayer* firstLayer);

	static bool MemoryEquals(const LayerMemoryDescription& right,
			const LayerMemoryDescription& left);
	static bool DataEquals(const LayerDataDescription& right,
			const LayerDataDescription& left);

};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNN_INTERLOCKHELPER_H_ */
