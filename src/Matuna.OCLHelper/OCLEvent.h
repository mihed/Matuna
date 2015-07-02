/*
* OCLEvent.h
*
*  Created on: Jul 1, 2015
*      Author: Mikael
*/

#ifndef MATUNA_MATUNA_OCLHELPER_OCLEVENT_H_
#define MATUNA_MATUNA_OCLHELPER_OCLEVENT_H_

#include "OCLInclude.h"

namespace Matuna
{
	namespace Helper
	{

		//TODO: This class should be movable, or work like a uniuqe_ptrk
		class OCLEvent
		{

		private:
			cl_event oclEvent;

		public:
			OCLEvent(cl_event oclEvent);
			OCLEvent(OCLEvent&& otherEvent);
			~OCLEvent();

			cl_event GetCLEvent();

			void Wait();

			cl_ulong GetProfilingInfo(cl_profiling_info info);
		};

	} /* namespace Helper */
} /* namespace Matuna */

#endif /* MATUNA_MATUNA_OCLHELPER_OCLEVENT_H_ */
