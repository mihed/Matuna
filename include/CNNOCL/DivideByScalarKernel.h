/*
 * DivideByScalarKernel.h
 *
 *  Created on: May 19, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_CNNOCL_DIVIDEBYSCALARKERNEL_H_
#define MATUNA_CNNOCL_DIVIDEBYSCALARKERNEL_H_

#include "OCLHelper/OCLKernelProgram.h"
#include "OCLHelper/OCLMemory.h"
#include <string>
#include <vector>

using namespace Matuna::Helper;
using namespace std;

namespace Matuna
{
namespace MachineLearning
{

template<class T>
class DivideByScalarKernel: public OCLKernelProgram
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

	void SetInputOutput(OCLMemory* inputOutput);
	void SetScalar(OCLMemory* scalar);

	virtual string ProgramName() const override;
	virtual string GetCompilerOptions() const override;
	virtual vector<string> GetProgramCode() const override;
	virtual string KernelName() const override;
	virtual const vector<size_t>& GlobalWorkSize() const override;
	virtual const vector<size_t>& LocalWorkSize() const override;
};

} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_CNNOCL_DIVIDEBYSCALARKERNEL_H_ */
