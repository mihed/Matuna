/*
* MaxPoolingLayer.cpp
*
*  Created on: Jun 23, 2015
*      Author: Mikael
*/

#include "MaxPoolingLayer.h"

#include "CheckPrecision.h"

#include "Matuna.Helper/Path.h"
#include "Matuna.Helper/FileHelper.h"
#include "Matuna.Helper/Converter.h"

namespace Matuna
{
	namespace MachineLearning
	{

		template<class T>
		MaxPoolingLayer<T>::MaxPoolingLayer(shared_ptr<OCLContext> context,
			const vector<LayerDataDescription>& inputLayerDescriptions,
			MatunaActivationFunction backPropActivation,
			const MaxPoolingLayerConfig* config)
			:
		OCLForwardBackPropLayer<T>(context, inputLayerDescriptions,
			backPropActivation, config), config(*config)
		{
			if (inputLayerDescriptions.size() == 0)
				throw invalid_argument(
				"There's no input data descriptions for the convolution layer.");

			if (inputLayerDescriptions.size() != 1)
				throw runtime_error("Not implemented exception");

			//Make sure the type we want to execute is supported on the device.
			static_assert(is_same<cl_double, T>::value || is_same<cl_float, T>::value, "The type is not yet supported");
			for (auto device : context->GetDevices()) 
			{
				auto deviceInfo = device->DeviceInfo();
				CheckPrecision<is_same<cl_double, T>::value>::Check(deviceInfo);
			}

			InitializeMemoryDescriptions(inputLayerDescriptions, config);
		}

		template<class T>
		MaxPoolingLayer<T>::~MaxPoolingLayer()
		{

		}

		template<class T>
		void MaxPoolingLayer<T>::InitializeMemoryDescriptions(const vector<LayerDataDescription>& inputLayerDescriptions, const MaxPoolingLayerConfig* config)
		{
			int sampligSizeWidth = config->SamplingSizeWidth();
			int sampligSizeHeight = config->SamplingSizeHeight();
			for (auto& layerDescription : inputLayerDescriptions)
			{
				LayerMemoryDescription inForwardMemoryProp;
				inForwardMemoryProp.Height = layerDescription.Height;
				inForwardMemoryProp.Width = layerDescription.Width;
				inForwardMemoryProp.Units = layerDescription.Units;
				inForwardMemoryProp.HeightOffset = 0;
				inForwardMemoryProp.WidthOffset = 0;
				inForwardMemoryProp.UnitOffset = 0;

				this->inForwardPropMemoryProposals.push_back(inForwardMemoryProp);
				this->outBackPropMemoryProposals.push_back(inForwardMemoryProp);

				int outDataHeight;
				if ((layerDescription.Height % sampligSizeHeight) == 0)
					outDataHeight = layerDescription.Height / sampligSizeHeight;
				else //We should have a log of some sort where we could emit warnings
					outDataHeight = static_cast<int>(floor(double(layerDescription.Height) / sampligSizeHeight));

				outDataHeight = outDataHeight == 0 ? 1 : outDataHeight; //Purely a safetly precaution for user's that don't calculate the diemensions

				int outDataWidth;
				if ((layerDescription.Width % sampligSizeWidth) == 0)
					outDataWidth = layerDescription.Width / sampligSizeWidth;
				else //We should have a log of some sort where we could emit warnings
					outDataWidth = static_cast<int>(floor(double(layerDescription.Width) / sampligSizeWidth));

				outDataWidth = outDataWidth == 0 ? 1 : outDataWidth; //Purely a safetly precaution for user's that don't calculate the diemensions

				LayerDataDescription outForwardDataDesc;
				outForwardDataDesc.Height = outDataHeight;
				outForwardDataDesc.Width = outDataWidth;
				outForwardDataDesc.Units = layerDescription.Units;
				this->outForwardPropDataDescriptions.push_back(outForwardDataDesc);

				LayerMemoryDescription outForwardMemoryProp;
				outForwardMemoryProp.Height = outDataHeight;
				outForwardMemoryProp.Width = outDataWidth;
				outForwardMemoryProp.Units = layerDescription.Units;
				outForwardMemoryProp.UnitOffset = 0;
				outForwardMemoryProp.WidthOffset = 0;
				outForwardMemoryProp.HeightOffset = 0;

				this->outForwardPropMemoryProposals.push_back(outForwardMemoryProp);
				this->inBackPropMemoryProposals.push_back(outForwardMemoryProp);
			}

			this->inBackPropDataDescriptions = this->outForwardPropDataDescriptions;
		}

		template<class T>
		MaxPoolingLayerConfig MaxPoolingLayer<T>::GetConfig() const
		{
			return config;
		}

		template<class T>
		void MaxPoolingLayer<T>::InterlockFinalized()
		{
			InitializePrograms();
		}

		template<class T>
		void MaxPoolingLayer<T>::InitializePrograms()
		{
			vector<OCLDevice*> devices = this->context->GetDevices();
			for (auto device : devices)
			{
				unique_ptr<OCLProgram> program(new OCLProgram());

				program->AddIncludePath(OCLProgram::DefaultSourceLocation);

				program->SetUseRelaxedMath(config.UseRelaxedMath());
				if (is_same<cl_double, T>::value) 
					program->AddDefine("DOUBLE_PRECISION");

				auto backActivationFunction = this->BackPropActivationFunction();
				if (backActivationFunction == MatunaSigmoidActivation)
					program->AddDefine("MATUNA_ACTIVATION_DERIVATIVE_SIGMOID");
				else if (backActivationFunction == MatunaTanhActivation)
					program->AddDefine("MATUNA_ACTIVATION_DERIVATIVE_TANH");
				else if (backActivationFunction == MatunaSoftMaxActivation)
					throw invalid_argument("Softmax is not allowed in a convolution layer at the moment");

				program->SetName("MaxPoolingSamplingProgram" + Converter::ConvertToString(program->InstanceCount()));

				InitializeMaxPoolingSamplingKernel(device, program.get());

				vector<OCLDevice*> oneDeviceVector;
				oneDeviceVector.push_back(device);
				this->context->AttachProgram(move(program), oneDeviceVector);
			}

			//TODO: At the moment we cache the indices even though it's necessarily back propped
			LayerDataDescription outputDataDesc = this->OutForwardPropDataDescriptions()[0];
			xMaxIndices = this->context->CreateMemory(CL_MEM_READ_WRITE,
				sizeof(cl_int) * outputDataDesc.Width);
			yMaxIndices = this->context->CreateMemory(CL_MEM_READ_WRITE,
				sizeof(cl_int) * outputDataDesc.Height);
		}

		template<class T>
		void MaxPoolingLayer<T>::InitializeMaxPoolingSamplingKernel(OCLDevice* device, OCLProgram* program)
		{
			string maxPoolingSamplingKernelPath = Path::Combine(OCLProgram::DefaultSourceLocation, "MaxPoolingSamplingKernel.cl");

			LayerDataDescription outputDataDesc = this->OutForwardPropDataDescriptions()[0];
			LayerMemoryDescription inForwardMemoryDesc = this->InForwardPropMemoryDescriptions()[0];
			LayerDataDescription inForwardDataDesc = this->InForwardPropDataDescriptions()[0];
			LayerMemoryDescription outForwardMemoryDesc = this->OutForwardPropMemoryDescriptions()[0];

			auto deviceInfo = device->DeviceInfo();
			auto maximumConstantBufferSize = deviceInfo.MaxConstantBufferSize();

			unique_ptr<LayerKernel<T>> kernel(new LayerKernel<T>());
			kernel->AddSourcePath(maxPoolingSamplingKernelPath);
			kernel->AddIncludePath(OCLProgram::DefaultSourceLocation);
			kernel->SetKernelName("MaxPoolingSamplingKernel");

			auto byteSize = inForwardMemoryDesc.TotalMemory() * sizeof(T);
			if (maximumConstantBufferSize > byteSize)
			{
				kernel->AddDefine(maxPoolingSamplingKernelPath, "CONSTANT_INPUT");
				maximumConstantBufferSize -= byteSize;
			}

			kernel->AddGlobalSize(outputDataDesc.Width);
			kernel->AddGlobalSize(outputDataDesc.Height);
			kernel->AddGlobalSize(outputDataDesc.Units);

			kernel->AddDefineSubsitute(maxPoolingSamplingKernelPath, "SAMPLING_SIZE_WIDTH", config.SamplingSizeWidth());
			kernel->AddDefineSubsitute(maxPoolingSamplingKernelPath, "SAMPLING_SIZE_HEIGHT", config.SamplingSizeHeight());
			kernel->AddDefineSubsitute(maxPoolingSamplingKernelPath, "INPUT_UNIT_MEMORY_WIDTH_OFFSET", inForwardMemoryDesc.WidthOffset);
			kernel->AddDefineSubsitute(maxPoolingSamplingKernelPath, "INPUT_UNIT_MEMORY_HEIGHT_OFFSET", inForwardMemoryDesc.HeightOffset);
			kernel->AddDefineSubsitute(maxPoolingSamplingKernelPath, "OUTPUT_UNIT_MEMORY_WIDTH_OFFSET", outForwardMemoryDesc.WidthOffset);
			kernel->AddDefineSubsitute(maxPoolingSamplingKernelPath, "OUTPUT_UNIT_MEMORY_HEIGHT_OFFSET", outForwardMemoryDesc.HeightOffset);
			kernel->AddDefineSubsitute(maxPoolingSamplingKernelPath, "OUTPUT_UNIT_OFFSET", outForwardMemoryDesc.UnitOffset);
			kernel->AddDefineSubsitute(maxPoolingSamplingKernelPath, "INPUT_UNIT_OFFSET", inForwardMemoryDesc.UnitOffset);
			kernel->AddDefineSubsitute(maxPoolingSamplingKernelPath, "OUTPUT_UNIT_MEMORY_WIDTH", outForwardMemoryDesc.Width);
			kernel->AddDefineSubsitute(maxPoolingSamplingKernelPath, "INPUT_UNIT_MEMORY_WIDTH", inForwardMemoryDesc.Width);
			kernel->AddDefineSubsitute(maxPoolingSamplingKernelPath, "OUTPUT_UNIT_MEMORY_ELEMENTS", outForwardMemoryDesc.Width * outForwardMemoryDesc.Height);
			kernel->AddDefineSubsitute(maxPoolingSamplingKernelPath, "INPUT_UNIT_MEMORY_ELEMENTS", inForwardMemoryDesc.Width * inForwardMemoryDesc.Height);

			//Determine whether or not we have to use a remainder
			int widthRemainder = inForwardDataDesc.Width % config.SamplingSizeWidth();
			int heightRemainder = inForwardDataDesc.Height % config.SamplingSizeHeight();
			if (widthRemainder != 0 || heightRemainder != 0)
			{
				kernel->AddDefine(maxPoolingSamplingKernelPath, "USE_REMAINDER");
				kernel->AddDefineSubsitute(maxPoolingSamplingKernelPath, "REMAINDER_SIZE_WIDTH", widthRemainder);
				kernel->AddDefineSubsitute(maxPoolingSamplingKernelPath, "REMAINDER_SIZE_HEIGHT", heightRemainder);
				kernel->AddDefineSubsitute(maxPoolingSamplingKernelPath, "X_MAX_INDEX", outputDataDesc.Width);
				kernel->AddDefineSubsitute(maxPoolingSamplingKernelPath, "Y_MAX_INDEX", outputDataDesc.Height);
			}

			deviceAndMaxPoolingSamplingKernels.insert(make_pair(device, kernel.get()));
			program->AttachKernel(move(kernel));
		}

		template<class T>
		void MaxPoolingLayer<T>::EnqueueForwardPropagation(OCLDevice* device, int queueIndex,
			OCLMemory* previousInput, OCLMemory* output, bool blocking)
		{
			//TODO: At the moment we cache the indices even though it's not necessarily back propped
			auto kernel = deviceAndMaxPoolingSamplingKernels[device];
			kernel->SetMemoryArg(previousInput, 0);
			kernel->SetMemoryArg(output, 1);
			kernel->SetMemoryArg(xMaxIndices.get(), 2);
			kernel->SetMemoryArg(yMaxIndices.get(), 3);
			device->ExecuteKernel(kernel, queueIndex, blocking);
		}

		template<class T>
		void MaxPoolingLayer<T>::EnqueueBackPropagation(OCLDevice* device, int queueIndex,
			OCLMemory* previousInput, OCLMemory* delta,
			OCLMemory* deltaOutput, bool blocking) 
		{

		}

		template<class T>
		vector<OCLMemory*> MaxPoolingLayer<T>::GetParameters()
		{
			return vector<OCLMemory*>();
		}

		template<class T>
		void MaxPoolingLayer<T>::GetParameters(T*, OCLDevice*,
			int, bool)
		{

		}

		template<class T>
		void MaxPoolingLayer<T>::SetParameters(T*, OCLDevice*,
			int, bool)
		{

		}

		template<class T>
		void MaxPoolingLayer<T>::EnqueueCalculateGradient(OCLDevice*, int,
			OCLMemory*, OCLMemory*, vector<OCLMemory*>, bool)
		{

		}

		template<class T>
		vector<size_t> MaxPoolingLayer<T>::GetMultipleParameterCount()
		{
			return vector<size_t>();
		}

		template<class T>
		size_t MaxPoolingLayer<T>::GetParameterCount()
		{
			return 0;
		}

		template class MaxPoolingLayer<cl_float>;
		template class MaxPoolingLayer<cl_double>;

	} /* namespace MachineLearning */
} /* namespace Matuna */
