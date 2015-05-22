/*
 * ConvolutionKernel.h
 *
 *  Created on: May 20, 2015
 *      Author: Mikael
 */

#ifndef ATML_CNNOPENCL_CONVOLUTIONKERNEL_H_
#define ATML_CNNOPENCL_CONVOLUTIONKERNEL_H_

#include "OpenCLHelper/OpenCLKernelProgram.h"
#include "CNN/ATMLActivationFunctionEnum.h"
#include "CNN/ATMLComputationPrecision.h"
#include "OpenCLHelper/OpenCLMemory.h"

using namespace ATML::Helper;
using namespace std;

namespace ATML
{
	namespace MachineLearning
	{

		template<class T>
		class ConvolutionKernel : public OpenCLKernelProgram
		{
		private:
			vector<size_t> globalWorkSize;
			vector<size_t> localWorkSize;
			string kernelName;
			string programName;
			string compilerOptions;

			bool useConstantInput;
			bool useConstantFilters;
			bool useConstantBias;
			bool useLocalMemory;
			bool useRelaxedMath;

			ATMLActivationFunction activation;
			ATMLComputationPrecision precision;

			int filterWidth; 
			int filterHeight;  
			int inputOffsetWidth;
			int inputOffsetHeight; 
			int outputOffsetWidth; 
			int outputOffsetHeight; 
			int outputOffsetUnit; 
			int outputStride; 
			int inputStride;
			int outputUnitMemoryCount; 
			int filterUnitElementCount;

			int dataOutputUnits;
			int dataOutputWidth;
			int dataOutputHeight;

		public:
			ConvolutionKernel(int dataOutputUnits, int dataOutputWidth, int dataOutputHeight, int filterWidth, int filterHeight, int inputOffsetWidth,
				int inputOffsetHeight, int outputOffsetWidth, int outputOffsetHeight, int outputOffsetUnit, int outputStride, int inputStride,
				int outputUnitMemoryCount, int filterUnitElementCount, bool useLocalMemory = false);
			~ConvolutionKernel();

			void SetInput(OpenCLMemory* input);
			void SetOutput(OpenCLMemory* output);
			void SetBiases(OpenCLMemory* biases);
			void SetFilters(OpenCLMemory* filters);

			void SetLocalWorkGroup(int width, int height);

			void SetConstantInput(bool value);
			void SetConstantFilters(bool value);
			void SetConstantBias(bool value);
			void SetRelaxedMath(bool value);
			void SetActivationFunction(ATMLActivationFunction activation);
			void SetComputationPrecision(ATMLComputationPrecision precision);

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

#endif /* ATML_CNNOPENCL_CONVOLUTIONKERNEL_H_ */