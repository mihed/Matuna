/*
 * ImageForwardPerceptronKernel.h
 *
 *  Created on: Jun 2, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_OCLConvNet_IMAGEFORWARDPERCEPTRONKERNEL_H_
#define MATUNA_OCLConvNet_IMAGEFORWARDPERCEPTRONKERNEL_H_

#include "OCLHelper/OCLKernelProgram.h"
#include "OCLHelper/OCLMemory.h"
#include "ConvNet/MatunaActivationFunctionEnum.h"
#include "ConvNet/MatunaComputationPrecision.h"

using namespace Matuna::Helper;
using namespace std;

namespace Matuna
{
namespace MachineLearning
{

template<class T>
class ImageForwardPerceptronKernel: public OCLKernelProgram
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

	MatunaActivationFunction activationFunction;
	MatunaComputationPrecision computationPrecision;

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
	void SetActivationFunction(MatunaActivationFunction activationFunction);
	void SetComputationPrecision(MatunaComputationPrecision computationPrecision);
	void SetWeights(OCLMemory* weights);
	void SetBiases(OCLMemory* biases);

	//Changed for every execution
	void SetInput(OCLMemory* input);
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

#endif /* MATUNA_OCLConvNet_IMAGEFORWARDPERCEPTRONKERNEL_H_ */
