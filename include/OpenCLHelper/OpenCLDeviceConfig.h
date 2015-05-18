/*
 * OpenCLDeviceConfig.h
 *
 *  Created on: May 6, 2015
 *      Author: Mikael
 */

#ifndef ATML_OPENCLHELPER_OPENCLDEVICECONFIG_H_
#define ATML_OPENCLHELPER_OPENCLDEVICECONFIG_H_

#include "OpenCLInclude.h"
#include <vector>

using namespace std;

namespace ATML
{
namespace Helper
{

class OpenCLDeviceConfig
{
private:
	vector<cl_command_queue_properties> properties;

public:
	OpenCLDeviceConfig()
	{
	}
	;
	~OpenCLDeviceConfig()
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
} /* namespace ATML */

#endif /* ATML_OPENCLHELPER_OPENCLDEVICECONFIG_H_ */
