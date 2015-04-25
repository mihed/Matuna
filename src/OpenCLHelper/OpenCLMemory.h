#ifndef CL_CL_H_ 
#define CL_CL_H_
#include <CL\cl.h>
#endif 

namespace ATML
{
	namespace Helper
	{
		//Forward declarations to avoid circular includes
		class OpenCLDevice;

		//This class 
		class OpenCLMemory
		{
			friend class OpenCLDevice;

		public:
			const cl_mem_flags readWriteFlag;

		private:
			const cl_mem memory;

			//Purpose of this pointer is only to make sure that the correct memory is given to the correct device
			//It doesn't take any ownership and is completely insensitive to the deletion of the owning device. (SAFE USE)

			//TODO: In the future, this could be replaced by the context since we can share memory between devices
			const OpenCLDevice* const owningDevice;
		public:
			OpenCLMemory(cl_mem memory, const OpenCLDevice* const owningDevice, cl_mem_flags readWriteFlag);
			~OpenCLMemory();
		};
	}
}

