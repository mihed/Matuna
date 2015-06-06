/*
 * BackPerceptronKernel.h
 *
 *  Created on: May 16, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_OCLConvNet_BACKPERCEPTRONKERNEL_H_
#define MATUNA_OCLConvNet_BACKPERCEPTRONKERNEL_H_

#include "OCLHelper/OCLKernelProgram.h"
#include "ConvNet/MatunaActivationFunctionEnum.h"
#include "OCLHelper/OCLMemory.h"

using namespace Matuna::Helper;

namespace Matuna
{
namespace MachineLearning
{

template<class T>
class BackPerceptronKernel: public OCLKernelProgram
{
private:
	OCLMemory* weights;
	vector<size_t> globalWorkSize;
	vector<size_t> localWorkSize;
	string kernelName;
	string programName;
	string compilerOptions;
	MatunaActivationFunction activationFunction;
	bool useRelaxedMath;
	bool useConstantInput;
	bool useConstantDeltaInput;
	bool useConstantWeights;
	int inputUnits;
	int outputUnits;
	int inputDeltaOffset;
	int inputOffset;
	int outputOffset;
public:
	BackPerceptronKernel(int inputUnits, int outputUnits, int inputDeltaOffset,
			int inputOffset, int outputOffset);
	~BackPerceptronKernel();

	void SetUseConstantWeights(bool value);
	void SetUseConstantInput(bool value);
	void SetUseConstantDeltaInput(bool value);
	void SetUseRelaxedMath(bool value);
	void SetActivationFunction(MatunaActivationFunction activationFunction);
	void SetWeights(OCLMemory* weights);

	//Changed for every execution
	void SetInput(OCLMemory* input);
	//Changed for every execution
	void SetDeltaInput(OCLMemory* deltaInput);
	//Changed for every execution
	void SetOutput(OCLMemory* output);

	void InitializeArguments();
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

#endif /* MATUNA_OCLConvNet_BACKPERCEPTRONKERNEL_H_ */
