/*
 * ConvolutionKernel.h
 *
 *  Created on: May 20, 2015
 *      Author: Mikael
 */

#ifndef MATUNA_ConvNetOCL_CONVOLUTIONKERNEL_H_
#define MATUNA_ConvNetOCL_CONVOLUTIONKERNEL_H_

#include "OCLHelper/OCLKernelProgram.h"
#include "ConvNet/MatunaActivationFunctionEnum.h"
#include "ConvNet/MatunaComputationPrecision.h"
#include "OCLHelper/OCLMemory.h"

using namespace Matuna::Helper;
using namespace std;

namespace Matuna
{
	namespace MachineLearning
	{

		template<class T>
		class ConvolutionKernel : public OCLKernelProgram
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

			MatunaActivationFunction activation;
			MatunaComputationPrecision precision;

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

			void SetInput(OCLMemory* input);
			void SetOutput(OCLMemory* output);
			void SetBiases(OCLMemory* biases);
			void SetFilters(OCLMemory* filters);

			void SetLocalWorkGroup(int width, int height);

			void SetConstantInput(bool value);
			void SetConstantFilters(bool value);
			void SetConstantBias(bool value);
			void SetRelaxedMath(bool value);
			void SetActivationFunction(MatunaActivationFunction activation);
			void SetComputationPrecision(MatunaComputationPrecision precision);

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

#endif /* MATUNA_ConvNetOCL_CONVOLUTIONKERNEL_H_ */
