/*
 * InterlockHelper.h
 *
 *  Created on: May 3, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_MATUNA_CONVNET_INTERLOCKHELPER_H_
#define MATUNA_MATUNA_CONVNET_INTERLOCKHELPER_H_

#include <vector>

using namespace std;

namespace Matuna
{
namespace MachineLearning
{

//Forward declarations
class LayerMemoryDescription;
class LayerDataDescription;

class InterlockHelper
{
public:
	static LayerMemoryDescription CalculateCompatibleMemory(
			const LayerMemoryDescription& right,
			const LayerMemoryDescription& left);

	static vector<LayerMemoryDescription> CalculateCompatibleMemory(
			const vector<LayerMemoryDescription>& right,
			const vector<LayerMemoryDescription>& left);

	static bool IsCompatible(const LayerMemoryDescription& right,
			const LayerMemoryDescription& left);

	static bool IsCompatible(const vector<LayerMemoryDescription>& right,
			const vector<LayerMemoryDescription>& left);

	static bool IsCompatible(const LayerDataDescription& data,
			const LayerMemoryDescription& memory);

	static bool IsCompatible(const vector<LayerDataDescription>& data,
			const vector<LayerMemoryDescription>& memory);


	static bool MemoryEquals(const LayerMemoryDescription& right,
			const LayerMemoryDescription& left);

	static bool MemoryEquals(const vector<LayerMemoryDescription>& right,
			const vector<LayerMemoryDescription>& left);


	static bool DataEquals(const LayerDataDescription& right,
			const LayerDataDescription& left);

	static bool DataEquals(const vector<LayerDataDescription>& right,
			const vector<LayerDataDescription>& left);

};

} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_MATUNA_CONVNET_INTERLOCKHELPER_H_ */
