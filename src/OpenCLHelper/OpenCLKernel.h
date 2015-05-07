/*
 * OpenCLKernel.h
 *
 *  Created on: Apr 28, 2015
 *      Author: Mikael
 */

#ifndef ATML_OPENCLHELPER_OPENCLKERNEL_H_
#define ATML_OPENCLHELPER_OPENCLKERNEL_H_

#include <CL/cl.h>
#include <string>
#include <vector>

using namespace std;

namespace ATML
{
namespace Helper
{

class OpenCLContext;

/**
 *@brief This class serves as an abstract base for every OpenCLKernel that are to be executed on the OpenCLDevice.
 */
class OpenCLKernel
{
	friend class OpenCLContext;

private:
	bool kernelSet;
	const OpenCLContext* context;

protected:
	static int instanceCounter;
	bool argumentsSet;
	cl_kernel kernel;

private:
	//This function is called by the OpenCLContext when created
	void SetOCLKernel(cl_kernel kernel);

	//This function is called by the OpenCLContext when created
	void SetContext(const OpenCLContext* const context);

public:
	OpenCLKernel();
	virtual ~OpenCLKernel();

	bool KernelSet() const
	{
		return kernelSet;
	}

	bool ArgumentsSet() const
	{
		return argumentsSet;
	}
	;

	bool ContextSet() const
	{
		return context ? true : false;
	}
	;

	const OpenCLContext* const GetContext() const
	{
		return context;
	}
	;

	cl_kernel GetKernel() const
	{
		return kernel;
	}
	;

	virtual void SetArguments() = 0;

	/**
	 *@brief The name of the program that has the implementation of this kernel
	 *
	 *Every program name is different as long as their program code is different.
	 *
	 *@return a string representing the program name.
	 */
	virtual string ProgramName() const = 0;

	/**
	 *@brief The name of the kernel that this OpenCLKernel represents
	 *@return a string containing the kernel name.
	 */
	virtual string KernelName() const = 0;

	/**
	 *@brief Gets the global work size of the OpenCLKernel
	 *
	 *Every OpenCL kernel call has a global work size that defines the dimension of the kernel call.
	 *This function returns the amount of dimensions and the size in every dimension of the kernel call.
	 *
	 *@return A constant vector reference with the global work size dimensions.
	 *@warning the reference is ONLY valid as long as the OpenCLKernel is alive.
	 */
	virtual const vector<size_t>& GlobalWorkSize() const = 0;

	/**
	 *@brief Gets the local work size of the OpenCLKernel
	 *
	 *Every OpenCl kernel call has a global and a local work size.
	 *When a kernel is executed on a device, the local work group is executed on the same compute unit
	 *and may have synchronization operations inside this group.
	 *
	 *@return A constant vector reference with the local work size dimensions.
	 *@warning the reference is ONLY valid as long as the OpenCLKernel is alive.
	 */
	virtual const vector<size_t>& LocalWorkSize() const = 0;
};

}
/* namespace Helper */
} /* namespace ATML */

#endif /* ATML_OPENCLHELPER_OPENCLKERNEL_H_ */
