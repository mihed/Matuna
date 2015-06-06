/*
 * GradientPerceptronKernel.h
 *
 *  Created on: May 17, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_ConvNetOCL_GRADIENTPERCEPTRONKERNEL_H_
#define MATUNA_ConvNetOCL_GRADIENTPERCEPTRONKERNEL_H_

#include "OCLHelper/OCLKernelProgram.h"
#include "OCLHelper/OCLMemory.h"

using namespace Matuna::Helper;

namespace Matuna
{
namespace MachineLearning
{

template<class T>
class GradientPerceptronKernel: public OCLKernelProgram
{
private:
	vector<size_t> globalWorkSize;
	vector<size_t> localWorkSize;
	string kernelName;
	string programName;
	string compilerOptions;

	bool useConstantInput;
	bool useConstantInputDelta;
	bool useRelaxedMath;

	int inputUnits;
	int units;

public:
	GradientPerceptronKernel(int inputUnits, int units);
	~GradientPerceptronKernel();

	void SetConstantInput(bool value);
	void SetConstantInputDelta(bool value);
	void SetUseRelaxedMath(bool value);

	//Changed for every execution
	void SetInput(OCLMemory* input);
	//Changed for every execution
	void SetInputDelta(OCLMemory* inputDelta);
	//Changed for every execution
	void SetGradient(OCLMemory* gradient);

	void InitializeCompilerOptions();

	virtual string ProgramName() const override;
	virtual string GetCompilerOptions() const override;
	virtual vector<string> GetProgramCode() const override;
	virtual string KernelName() const override;
	virtual const vector<size_t>& GlobalWorkSize() const override;
	virtual const vector<size_t>& LocalWorkSize() const override;
};

} /* namespace MachineLearning */
} /* namespace Matuna */

#endif /* MATUNA_ConvNetOCL_GRADIENTPERCEPTRONKERNEL_H_ */
