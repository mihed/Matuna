/*
 * SimpleSumKernel.h
 *
 *  Created on: May 19, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_CNNOPENCL_SIMPLESUMKERNEL_H_
#define MATUNA_CNNOPENCL_SIMPLESUMKERNEL_H_

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

//Supposed to be executed as a TASK
template<class T>
class SimpleSumKernel: public OpenCLKernelProgram
{
private:
	vector<size_t> globalWorkSize;
	vector<size_t> localWorkSize;
	string kernelName;
	string programName;
	string compilerOptions;

	int inputCount;

public:
	SimpleSumKernel(int inputCount);
	~SimpleSumKernel();

	void SetInput(OpenCLMemory* input);
	void SetOutput(OpenCLMemory* output);

	virtual string ProgramName() const override;
	virtual string GetCompilerOptions() const override;
	virtual vector<string> GetProgramCode() const override;
	virtual string KernelName() const override;
	virtual const vector<size_t>& GlobalWorkSize() const override;
	virtual const vector<size_t>& LocalWorkSize() const override;
};

} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_CNNOPENCL_SIMPLESUMKERNEL_H_ */
