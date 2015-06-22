/*
* VanillaSamplingLayer.cpp
*
*  Created on: Jun 21, 2015
*      Author: Mikael
*/

#include "VanillaSamplingLayer.h"
#include "CheckPrecision.h"

#include "Matuna.Helper/Path.h"
#include "Matuna.Helper/FileHelper.h"
#include "Matuna.Helper/Converter.h"

namespace Matuna
{
	namespace MachineLearning
	{

		template<class T>
		VanillaSamplingLayer<T>::VanillaSamplingLayer(shared_ptr<OCLContext> context,
			const vector<LayerDataDescription>& inputLayerDescriptions,
			MatunaActivationFunction backPropActivation,
			const VanillaSamplingLayerConfig* config) :
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
		VanillaSamplingLayer<T>::~VanillaSamplingLayer()
		{

		}

		template<class T>
		void VanillaSamplingLayer<T>::InitializeMemoryDescriptions(const vector<LayerDataDescription>& inputLayerDescriptions, const VanillaSamplingLayerConfig* config)
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

				outDataHeight = outDataHeight == 0 ? 1 : outDataHeight;

				int outDataWidth;
				if ((layerDescription.Width % sampligSizeWidth) == 0)
					outDataWidth = layerDescription.Width / sampligSizeWidth;
				else //We should have a log of some sort where we could emit warnings
					outDataWidth = static_cast<int>(floor(double(layerDescription.Width) / sampligSizeWidth));

				outDataWidth = outDataWidth == 0 ? 1 : outDataWidth; 

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
		VanillaSamplingLayerConfig VanillaSamplingLayer<T>::GetConfig() const
		{
			return config;
		}

		template<class T>
		void VanillaSamplingLayer<T>::InterlockFinalized()
		{
			InitializePrograms();
		}

		template<class T>
		void VanillaSamplingLayer<T>::InitializePrograms()
		{
			vector<OCLDevice*> devices = this->context->GetDevices();
			for (auto device : devices)
			{
				unique_ptr<OCLProgram> program(new OCLProgram());

				program->AddIncludePath(OCLProgram::DefaultSourceLocation);

				if (is_same<cl_double, T>::value) 
					program->AddDefine("DOUBLE_PRECISION");

				program->SetName("VanillaSamlingProgram" + Converter::ConvertToString(program->InstanceCount()));

				InitializeVanillaSamplingKernel(device, program.get());

				vector<OCLDevice*> oneDeviceVector;
				oneDeviceVector.push_back(device);
				this->context->AttachProgram(move(program), oneDeviceVector);
			}
		}

		template<class T>
		void VanillaSamplingLayer<T>::InitializeVanillaSamplingKernel(OCLDevice* device, OCLProgram* program)
		{
			string vanillaSamplingKernelPath = Path::Combine(OCLProgram::DefaultSourceLocation, "VanillaSamplingKernel.cl");

			LayerDataDescription outputDataDesc = this->OutForwardPropDataDescriptions()[0];
			LayerMemoryDescription inForwardMemoryDesc = this->InForwardPropMemoryDescriptions()[0];
			LayerMemoryDescription outForwardMemoryDesc = this->OutForwardPropMemoryDescriptions()[0];

			auto deviceInfo = device->DeviceInfo();
			auto maximumConstantBufferSize = deviceInfo.MaxConstantBufferSize();

			unique_ptr<LayerKernel<T>> kernel(new LayerKernel<T>());
			kernel->AddSourcePath(vanillaSamplingKernelPath);
			kernel->AddIncludePath(OCLProgram::DefaultSourceLocation);
			kernel->SetKernelName("VanillaSamplingKernel");

			auto byteSize = inForwardMemoryDesc.TotalMemory() * sizeof(T);
			if (maximumConstantBufferSize > byteSize)
			{
				kernel->AddDefine(vanillaSamplingKernelPath, "CONSTANT_INPUT");
				maximumConstantBufferSize -= byteSize;
			}

			kernel->AddGlobalSize(outputDataDesc.Width);
			kernel->AddGlobalSize(outputDataDesc.Height);
			kernel->AddGlobalSize(outputDataDesc.Units);

			kernel->AddDefineSubsitute(vanillaSamplingKernelPath, "SAMPLING_SIZE_WIDTH", config.SamplingSizeWidth());
			kernel->AddDefineSubsitute(vanillaSamplingKernelPath, "SAMPLING_SIZE_HEIGHT", config.SamplingSizeHeight());
			kernel->AddDefineSubsitute(vanillaSamplingKernelPath, "INPUT_UNIT_MEMORY_WIDTH_OFFSET", inForwardMemoryDesc.WidthOffset);
			kernel->AddDefineSubsitute(vanillaSamplingKernelPath, "INPUT_UNIT_MEMORY_HEIGHT_OFFSET", inForwardMemoryDesc.HeightOffset);
			kernel->AddDefineSubsitute(vanillaSamplingKernelPath, "OUTPUT_UNIT_MEMORY_WIDTH_OFFSET", outForwardMemoryDesc.WidthOffset);
			kernel->AddDefineSubsitute(vanillaSamplingKernelPath, "OUTPUT_UNIT_MEMORY_HEIGHT_OFFSET", outForwardMemoryDesc.HeightOffset);
			kernel->AddDefineSubsitute(vanillaSamplingKernelPath, "OUTPUT_UNIT_OFFSET", outForwardMemoryDesc.UnitOffset);
			kernel->AddDefineSubsitute(vanillaSamplingKernelPath, "INPUT_UNIT_OFFSET", inForwardMemoryDesc.UnitOffset);
			kernel->AddDefineSubsitute(vanillaSamplingKernelPath, "OUTPUT_UNIT_MEMORY_WIDTH", outForwardMemoryDesc.Width);
			kernel->AddDefineSubsitute(vanillaSamplingKernelPath, "INPUT_UNIT_MEMORY_WIDTH", inForwardMemoryDesc.Width);
			kernel->AddDefineSubsitute(vanillaSamplingKernelPath, "OUTPUT_UNIT_MEMORY_ELEMENTS", outForwardMemoryDesc.Width * outForwardMemoryDesc.Height);
			kernel->AddDefineSubsitute(vanillaSamplingKernelPath, "INPUT_UNIT_MEMORY_ELEMENTS", inForwardMemoryDesc.Width * inForwardMemoryDesc.Height);

			deviceAndVanillaSamplingKernels.insert(make_pair(device, kernel.get()));
			program->AttachKernel(move(kernel));
		}

		template<class T>
		void VanillaSamplingLayer<T>::EnqueueForwardPropagation(OCLDevice* device, int queueIndex,
			OCLMemory* previousInput, OCLMemory* output, bool blocking)
		{
			auto kernel = deviceAndVanillaSamplingKernels[device];
			kernel->SetMemoryArg(previousInput, 0);
			kernel->SetMemoryArg(output, 1);
			device->ExecuteKernel(kernel, queueIndex, blocking);
		}

		template<class T>
		void VanillaSamplingLayer<T>::EnqueueBackPropagation(OCLDevice* device, int queueIndex,
			OCLMemory* previousInput, OCLMemory* delta,
			OCLMemory* deltaOutput, bool blocking)
		{

		}

		template<class T>
		vector<OCLMemory*> VanillaSamplingLayer<T>::GetParameters()
		{
			return vector<OCLMemory*>();
		}

		template<class T>
		void VanillaSamplingLayer<T>::GetParameters(T*, OCLDevice*,
			int, bool)
		{

		}

		template<class T>
		void VanillaSamplingLayer<T>::SetParameters(T*, OCLDevice*,
			int, bool)
		{

		}

		template<class T>
		void VanillaSamplingLayer<T>::EnqueueCalculateGradient(OCLDevice*, int,
			OCLMemory*, OCLMemory*, vector<OCLMemory*>, bool)
		{

		}

		template <class T>
		vector<size_t> VanillaSamplingLayer<T>::GetMultipleParameterCount()
		{
			return vector<size_t>();
		}

		template<class T>
		size_t VanillaSamplingLayer<T>::GetParameterCount()
		{
			return 0;
		}

		template class VanillaSamplingLayer<cl_float>;
		template class VanillaSamplingLayer<cl_double>;

	} /* namespace MachineLearning */
} /* namespace Matuna */
