/*
 * OCLDeviceConfig.h
 *
 *  Created on: May 6, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_MATUNA_OCLHELPER_OCLDEVICECONFIG_H_
#define MATUNA_MATUNA_OCLHELPER_OCLDEVICECONFIG_H_

#include "OCLInclude.h"
#include <vector>

using namespace std;

namespace Matuna
{
namespace Helper
{

class OCLDeviceConfig
{
private:
	vector<cl_command_queue_properties> properties;

public:
	OCLDeviceConfig()
	{
	}
	;
	~OCLDeviceConfig()
	{
	}
	;

	size_t CommandQueueCount() const
	{
		return properties.size();
	}
	;
	vector<cl_command_queue_properties> GetCommandQueues() const
	{
		return properties;
	}
	;

	void AddCommandQueue()
	{
		properties.push_back(0);
	}
	;

	void AddCommandQueue(cl_command_queue_properties property)
	{
		properties.push_back(property);
	}
	;
};

} /* namespace Helper */
} /* namespace Matuna */

#endif /* MATUNA_MATUNA_OCLHELPER_OCLDEVICECONFIG_H_ */
