/*
 * OCLKernel.h
 *
 *  Created on: Apr 28, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_OCLHELPER_OCLKERNEL_H_
#define MATUNA_OCLHELPER_OCLKERNEL_H_

#include "OCLInclude.h"
#include <string>
#include <vector>

using namespace std;

namespace Matuna
{
namespace Helper
{

class OCLContext;

/**
 *@brief This class serves as an abstract base for every OCLKernel that are to be executed on the OCLDevice.
 */
class OCLKernel
{
	friend class OCLContext;

private:
	bool kernelSet;
	const OCLContext* context;
	cl_kernel kernel;

protected:
	static int instanceCounter;

private:
	//This function is called by the OCLContext when created
	void SetOCLKernel(cl_kernel kernel);

	//This function is called by the OCLContext when created
	void SetContext(const OCLContext* const context);

public:
	OCLKernel();
	virtual ~OCLKernel();

	bool KernelSet() const
	{
		return kernelSet;
	}
	;

	bool ContextSet() const
	{
		return context ? true : false;
	}
	;

	const OCLContext* const GetContext() const
	{
		return context;
	}
	;

	cl_kernel GetKernel() const
	{
		return kernel;
	}
	;

	/**
	 *@brief The name of the program that has the implementation of this kernel
	 *
	 *Every program name is different as long as their program code is different.
	 *
	 *@return a string representing the program name.
	 */
	virtual string ProgramName() const = 0;

	/**
	 *@brief The name of the kernel that this OCLKernel represents
	 *@return a string containing the kernel name.
	 */
	virtual string KernelName() const = 0;

	/**
	 *@brief Gets the global work size of the OCLKernel
	 *
	 *Every OCL kernel call has a global work size that defines the dimension of the kernel call.
	 *This function returns the amount of dimensions and the size in every dimension of the kernel call.
	 *
	 *@return A constant vector reference with the global work size dimensions.
	 *@warning the reference is ONLY valid as long as the OCLKernel is alive.
	 */
	virtual const vector<size_t>& GlobalWorkSize() const = 0;

	/**
	 *@brief Gets the local work size of the OCLKernel
	 *
	 *Every OpenCl kernel call has a global and a local work size.
	 *When a kernel is executed on a device, the local work group is executed on the same compute unit
	 *and may have synchronization operations inside this group.
	 *
	 *@return A constant vector reference with the local work size dimensions.
	 *@warning the reference is ONLY valid as long as the OCLKernel is alive.
	 */
	virtual const vector<size_t>& LocalWorkSize() const = 0;
};

}
/* namespace Helper */
} /* namespace Matuna */

#endif /* MATUNA_OCLHELPER_OCLKERNEL_H_ */
