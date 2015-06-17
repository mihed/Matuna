/*
 * CheckPrecision.h
 *
 *  Created on: Jun 17, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_MATUNA_OCLCONVNET_CHECKPRECISION_H_
#define MATUNA_MATUNA_OCLCONVNET_CHECKPRECISION_H_

#include "Matuna.OCLHelper/OCLDeviceInfo.h"
#include <stdexcept>
using namespace std;
using namespace Matuna::Helper;

template<bool Condition>
struct CheckPrecision
{
};

//This specialization checks for double
template<>
struct CheckPrecision<true>
{
	static void Check(const OCLDeviceInfo& deviceInfo)
	{
		if (deviceInfo.PreferredDoubleVectorWidth() == 0)
			throw invalid_argument(
					"The template argument is not supported on the chosen devices");
	}
};

//This specialization checks for float
template<>
struct CheckPrecision<false>
{
	static void Check(const OCLDeviceInfo& deviceInfo)
	{
		if (deviceInfo.PreferredFloatVectorWidth() == 0)
			throw invalid_argument(
					"The template argument is not supported on the chosen devices");
	}
};

#endif /* SOURCE_DIRECTORY__MATUNA_OCLCONVNET_CHECKPRECISION_H_ */
