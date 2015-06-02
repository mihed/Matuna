/*
 * ImageForwardPerceptronKernel.h
 *
 *  Created on: Jun 2, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNNOPENCL_IMAGEFORWARDPERCEPTRONKERNEL_H_
#define ATML_CNNOPENCL_IMAGEFORWARDPERCEPTRONKERNEL_H_

#include "OpenCLHelper/OpenCLKernelProgram.h"
#include "OpenCLHelper/OpenCLMemory.h"
#include "CNN/ATMLActivationFunctionEnum.h"
#include "CNN/ATMLComputationPrecision.h"

using namespace ATML::Helper;
using namespace std;

namespace ATML
{
namespace MachineLearning
{

template<class T>
class ImageForwardPerceptronKernel: public OpenCLKernelProgram
{
private:
	vector<size_t> globalWorkSize;
	vector<size_t> localWorkSize;
	string kernelName;
	string programName;
	string compilerOptions;

	bool useConstantWeights;
	bool useConstantInput;
	bool useConstantBiases;
	bool useRelaxedMath;

	ATMLActivationFunction activationFunction;
	ATMLComputationPrecision computationPrecision;

	int globalUnits;
	int inputDataUnits;
	int inputDataWidth;
	int inputDataHeight;
	int inputWidthOffset;
	int inputHeightOffset;
	int inputUnitOffset;
	int inputStride;
	int inputMemoryHeight;
	int weightColumnCount;
	int outputUnitOffset;

public:
	ImageForwardPerceptronKernel(int globalUnits, int inputDataUnits,
			int inputDataWidth, int inputDataHeight, int inputWidthOffset,
			int inputHeightOffset, int inputUnitOffset, int inputStride,
			int inputMemoryHeight, int weightColumnCount, int outputUnitOffset);
	~ImageForwardPerceptronKernel();

	void SetUseConstantWeights(bool value);
	void SetUseConstantInput(bool value);
	void SetUseConstantBiases(bool value);
	void SetUseRelaxedMath(bool value);
	void SetActivationFunction(ATMLActivationFunction activationFunction);
	void SetComputationPrecision(ATMLComputationPrecision computationPrecision);
	void SetWeights(OpenCLMemory* weights);
	void SetBiases(OpenCLMemory* biases);

	//Changed for every execution
	void SetInput(OpenCLMemory* input);
	//Changed for every execution
	void SetOutput(OpenCLMemory* output);

	void InitializeCompilerOptions();

	virtual string ProgramName() const override;
	virtual string GetCompilerOptions() const override;
	virtual vector<string> GetProgramCode() const override;
	virtual string KernelName() const override;
	virtual const vector<size_t>& GlobalWorkSize() const override;
	virtual const vector<size_t>& LocalWorkSize() const override;

};

} /* namespace MachineLearning */
} /* namespace ATML */

#endif /* ATML_CNNOPENCL_IMAGEFORWARDPERCEPTRONKERNEL_H_ */
