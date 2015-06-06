/*
 * DivideByScalarKernel.h
 *
 *  Created on: May 19, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_CNNOPENCL_DIVIDEBYSCALARKERNEL_H_
#define MATUNA_CNNOPENCL_DIVIDEBYSCALARKERNEL_H_

#include "OpenCLHelper/OpenCLKernelProgram.h"
#include "OpenCLHelper/OpenCLMemory.h"
#include <string>
#include <vector>

using namespace Matuna::Helper;
using namespace std;

namespace Matuna
{
namespace MachineLearning
{

template<class T>
class DivideByScalarKernel: public OpenCLKernelProgram
{
private:
	vector<size_t> globalWorkSize;
	vector<size_t> localWorkSize;
	string kernelName;
	string programName;
	string compilerOptions;

	int inputCount;

public:
	DivideByScalarKernel(int inputCount);
	~DivideByScalarKernel();

	void SetInputOutput(OpenCLMemory* inputOutput);
	void SetScalar(OpenCLMemory* scalar);

	virtual string ProgramName() const override;
	virtual string GetCompilerOptions() const override;
	virtual vector<string> GetProgramCode() const override;
	virtual string KernelName() const override;
	virtual const vector<size_t>& GlobalWorkSize() const override;
	virtual const vector<size_t>& LocalWorkSize() const override;
};

} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_CNNOPENCL_DIVIDEBYSCALARKERNEL_H_ */
