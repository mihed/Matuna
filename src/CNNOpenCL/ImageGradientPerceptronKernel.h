/*
 * ImageGradientPerceptronKernel.h
 *
 *  Created on: Jun 3, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_CNNOPENCL_IMAGEGRADIENTPERCEPTRONKERNEL_H_
#define MATUNA_CNNOPENCL_IMAGEGRADIENTPERCEPTRONKERNEL_H_

#include "OpenCLHelper/OpenCLKernelProgram.h"
#include "OpenCLHelper/OpenCLMemory.h"

using namespace Matuna::Helper;

namespace Matuna
{
namespace MachineLearning
{

template<class T>
class ImageGradientPerceptronKernel: public OpenCLKernelProgram
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

	int weightWidth;
	int weightHeight;
	int inputDataWidth;
	int inputDataHeight;
	int inputWidthOffset;
	int inputHeightOffset;
	int inputUnitOffset;
	int inputStride;
	int inputMemoryHeight;
	int inputDeltaOffset;
	int weightColumnCount;

public:
	ImageGradientPerceptronKernel(int weightWidth, int weightHeight,
			int inputDataWidth, int inputDataHeight, int inputWidthOffset,
			int inputHeightOffset, int inputUnitOffset, int inputStride,
			int inputMemoryHeight, int inputDeltaOffset, int weightColumnCount);
	~ImageGradientPerceptronKernel();

	void SetConstantInput(bool value);
	void SetConstantInputDelta(bool value);
	void SetUseRelaxedMath(bool value);

	//Changed for every execution
	void SetInput(OpenCLMemory* input);
	//Changed for every execution
	void SetInputDelta(OpenCLMemory* inputDelta);
	//Changed for every execution
	void SetGradient(OpenCLMemory* gradient);

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

#endif /* MATUNA_CNNOPENCL_IMAGEGRADIENTPERCEPTRONKERNEL_H_ */
