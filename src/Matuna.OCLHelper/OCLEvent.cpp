/*
* OCLEvent.cpp
*
*  Created on: Jul 1, 2015
*      Author: Mikael
*/

#include "OCLEvent.h"
#include "OCLUtility.h"

namespace Matuna
{
	namespace Helper
	{

		OCLEvent::OCLEvent(cl_event oclEvent)
		{
			this->oclEvent = oclEvent;
		}

		OCLEvent::OCLEvent(OCLEvent&& otherEvent)
		{
			this->oclEvent = otherEvent.oclEvent;
			otherEvent.oclEvent = nullptr;
		}

		OCLEvent::~OCLEvent()
		{
			if (oclEvent != nullptr)
				CheckOCLError(clReleaseEvent(oclEvent), "The event could not be releasedk");
		}

		cl_event OCLEvent::GetCLEvent()
		{
			return oclEvent;
		}

		void OCLEvent::Wait()
		{
			CheckOCLError(clWaitForEvents(1, &oclEvent), "Could not wait for the event");
		}

		cl_ulong OCLEvent::GetProfilingInfo(cl_profiling_info info)
		{
			cl_ulong result;
			size_t returnedSize;
			CheckOCLError(clGetEventProfilingInfo(oclEvent, info, sizeof(result), &result, &returnedSize), "The profiling info could not be fetched");
			return result;
		}

	} /* namespace Helper */
} /* namespace Matuna */
