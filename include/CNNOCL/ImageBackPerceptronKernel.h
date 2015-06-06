/*
 * ImageBackPerceptronKernel.h
 *
 *  Created on: Jun 2, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_CNNOCL_IMAGEBACKPERCEPTRONKERNEL_H_
#define MATUNA_CNNOCL_IMAGEBACKPERCEPTRONKERNEL_H_

#include "OCLHelper/OCLKernelProgram.h"
#include "CNN/MatunaActivationFunctionEnum.h"
#include "OCLHelper/OCLMemory.h"

using namespace Matuna::Helper;
using namespace std;

namespace Matuna
{
namespace MachineLearning
{

template<class T>
class ImageBackPerceptronKernel: public OCLKernelProgram
{
private:
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

	int globalWidth;
	int globalHeight;
	int globalUnits;
	int outputWidthOffset;
	int outputHeightOffset;
	int outputUnitOffset;
	int inputWidthOffset;
	int inputHeightOffset;
	int inputUnitOffset;
	int outputStride;
	int outputMemoryHeight;
	int inputStride;
	int inputMemoryHeight;
	int weightColumnCount;
	int inputDeltaOffset;
	int inputDeltaDataUnits;

public:
	ImageBackPerceptronKernel(int globalWidth, int globalHeight,
			int globalUnits, int outputWidthOffset, int outputHeightOffset,
			int outputUnitOffset, int inputWidthOffset, int inputHeightOffset,
			int inputUnitOffset, int outputStride, int outputMemoryHeight,
			int inputStride, int inputMemoryHeight, int weightColumnCount,
			int inputDeltaOffset, int inputDeltaDataUnits);
	~ImageBackPerceptronKernel();

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

#endif /* MATUNA_CNNOCL_IMAGEBACKPERCEPTRONKERNEL_H_ */
