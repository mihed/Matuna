/*
 * OCLMemory.h
 *
 *  Created on: Apr 26, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_MATUNA_OCLHELPER_OCLMEMORY_H_
#define MATUNA_MATUNA_OCLHELPER_OCLMEMORY_H_

#include "OCLInclude.h"

namespace Matuna
{
namespace Helper
{

class OCLContext;

/**
 *@brief This class wrapps a native OCL memory pointer. Follows the RAII pattern, meaning that when the destructor is called the OCL memory is released.
 *
 *In this version of the library. The memory object is bound to the OCLDevice.
 *It must be used in conjunction to the OCLDevice that created the memory object.
 *In future version we can expect memory objects to be bound to the platform (context) instead.
 */
class OCLMemory
final
{
	private:
		cl_mem memory;
		cl_mem_flags readWriteFlag;
		OCLContext* owningContext;
		size_t byteSize;
	public:

		/**
		 *@brief Instantiate the OCLMemory. Normally called inside OCLDevice::CreateMemory
		 *
		 *The normal usage of this function is throug the OCLDevice.
		 */
		OCLMemory(cl_mem memory, OCLContext* owningContext,
				cl_mem_flags readWriteFlag, size_t byteSize);
		~OCLMemory();

		/**
		 *@brief The accessor of the memory: read, write or read-write.
		 *@return The OCL memory accessor flag of this memor.
		 */
		cl_mem_flags ReadWriteFlag() const;

		OCLContext* OwningContext() const;

		cl_mem GetCLMemory() const;

		size_t ByteSize() const;
	};

	} /* namespace Helper */
	} /* namespace Matuna */

#endif /* MATUNA_MATUNA_OCLHELPER_OCLMEMORY_H_ */
