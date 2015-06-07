/*
* StandardOutputLayer.cpp
*
*  Created on: May 5, 2015
*      Author: Mikael
*/

#include "StandardOutputLayer.h"
#include "Matuna.ConvNet/InterlockHelper.h"
#include <stdexcept>
#include <type_traits>

namespace Matuna {
	namespace MachineLearning {

		template<class T>
		StandardOutputLayer<T>::StandardOutputLayer(shared_ptr<OCLContext> context,
			const vector<LayerDataDescription>& inputLayerDescriptions,
			MatunaActivationFunction backPropActivation,
			const StandardOutputLayerConfig* outputLayerConfig) :
		OutputLayer(inputLayerDescriptions, backPropActivation,
			outputLayerConfig), context(context), config(*outputLayerConfig) {

				if (inputLayerDescriptions.size() == 0)
					throw invalid_argument(
					"There's no input data descriptions for the standard output layer.");

				//TODO: Make sure this works for a more general case later on.
				if (inputLayerDescriptions.size() > 1) {
					auto count = inputLayerDescriptions.size();
					for (int i = 1; i < count; i++)
						if (!InterlockHelper::DataEquals(inputLayerDescriptions[i - 1],
							inputLayerDescriptions[i]))
							throw invalid_argument(
							"We cannot have multiple different input descriptions for a standard output layer");
				}

				//The targets must have the same data descriptions as the inputs
				inBackPropDataDescriptions = inputLayerDescriptions;
				inputDescription = inputLayerDescriptions[0];

				for (auto& inputDescription : inputLayerDescriptions) {
					LayerMemoryDescription inBackPropMemProp;
					inBackPropMemProp.Height = inputDescription.Height;
					inBackPropMemProp.Width = inputDescription.Width;
					inBackPropMemProp.Units = inputDescription.Units;
					inBackPropMemProp.UnitOffset = 0;
					inBackPropMemProp.WidthOffset = 0;
					inBackPropMemProp.HeightOffset = 0;
					inBackPropMemoryProposals.push_back(inBackPropMemProp);
					outBackPropMemoryProposals.push_back(inBackPropMemProp);
					inForwardPropMemoryProposals.push_back(inBackPropMemProp);
				}

				useImage = false;
		}

		template<class T>
		StandardOutputLayer<T>::~StandardOutputLayer() {
			for (auto& deviceAndKernel : deviceAndOutputKernels) {
				auto& kernelProgram = deviceAndKernel.second;
				this->context->RemoveKernel(kernelProgram.get());
				this->context->RemoveProgram(kernelProgram.get());
			}

			for (auto& deviceAndKernel : deviceAndImageOutputKernels) {
				auto& kernelProgram = deviceAndKernel.second;
				this->context->RemoveKernel(kernelProgram.get());
				this->context->RemoveProgram(kernelProgram.get());
			}

			for (auto& deviceAndKernel : deviceAndErrorKernels) {
				auto& kernelProgram = deviceAndKernel.second;
				this->context->RemoveKernel(kernelProgram.get());
				this->context->RemoveProgram(kernelProgram.get());
			}

			for (auto& deviceAndKernel : deviceAndImageErrorKernels) {
				auto& kernelProgram = deviceAndKernel.second;
				this->context->RemoveKernel(kernelProgram.get());
				this->context->RemoveProgram(kernelProgram.get());
			}
		}

		template<class T>
		void StandardOutputLayer<T>::InterlockFinalized() {
			//TODO: make sure this layer is not limited to be connected to a perceptron

			auto inBackProp = inBackPropDataDescriptions[0];
			auto inBackPropMem = this->InBackPropMemoryDescriptions()[0];
			auto inForwardPropMem = this->InForwardPropMemoryDescriptions()[0];
			auto outBackPropMem = this->OutBackPropMemoryDescriptions()[0];

			if (!InterlockHelper::DataEquals(inputDescription, inBackProp))
				throw runtime_error("The targets are not the same as the inputs");

			if (!InterlockHelper::MemoryEquals(inBackPropMem, inForwardPropMem))
				throw runtime_error(
				"The inBackProp memory and the inForwardProp memory doesn't correspond");

			//Make sure the type we want to execute is supported on the device.
			vector<OCLDevice*> devices = this->context->GetDevices();
			for (auto device : devices) {
				auto deviceInfo = device->DeviceInfo();
				if (is_same<cl_double, T>::value) {
					if (deviceInfo.PreferredDoubleVectorWidth() == 0)
						throw invalid_argument(
						"The template argument is not supported on the chosen devices");
				} else if (is_same<cl_float, T>::value) {
					if (deviceInfo.PreferredFloatVectorWidth() == 0)
						throw invalid_argument(
						"The template argument is not supported on the chosen devices");
				} else
					throw runtime_error(
					"The template argument does not match the supported arguments");
			}

			if (outBackPropMem.Width != 1 || outBackPropMem.Height != 1)
			{
				InitializeImageOutputKernel();
				InitializeImageErrorKernel();
				useImage = true;
			}
			else
			{
				InitializeErrorKernel();
				InitializeOutputKernel();
			}
		}

		template<class T>
		void StandardOutputLayer<T>::InitializeErrorKernel() {
			auto inForwardPropMem = this->InForwardPropMemoryDescriptions()[0];
			vector<OCLDevice*> devices = this->context->GetDevices();
			for (auto device : devices) {
				auto deviceInfo = device->DeviceInfo();

				unique_ptr<ErrorKernel<T>> kernel(
					new ErrorKernel<T>(inputDescription.Units,
					inForwardPropMem.UnitOffset));

				auto maximumConstantBufferSize = deviceInfo.MaxConstantBufferSize();
				auto inputTargetBytes = sizeof(T) * inputDescription.Units;
				if (maximumConstantBufferSize > inputTargetBytes) {
					kernel->SetConstantInput(true);
					maximumConstantBufferSize -= inputTargetBytes;
				}
				if (maximumConstantBufferSize > inputTargetBytes) {
					kernel->SetConstantTarget(true);
					maximumConstantBufferSize -= inputTargetBytes;
				}

				kernel->SetUseRelaxedMath(config.UseRelaxedMath());
				kernel->SetComputationPrecision(config.ComputationPrecision());
				kernel->SetErrorFunction(config.ErrorFunction());
				kernel->InitializeCompilerOptions();
				vector<OCLDevice*> oneDeviceVector;
				oneDeviceVector.push_back(device);
				this->context->AddProgramFromSource(kernel.get(), oneDeviceVector);
				this->context->AddKernel(kernel.get());
				deviceAndErrorKernels.insert(make_pair(device, move(kernel)));
			}
		}

		template<class T>
		void StandardOutputLayer<T>::InitializeImageErrorKernel()
		{
			auto inForwardPropMem = this->InForwardPropMemoryDescriptions()[0];
			auto inForwardPropData = this->InForwardPropDataDescriptions()[0];
			vector<OCLDevice*> devices = this->context->GetDevices();
			for (auto device : devices) {
				auto deviceInfo = device->DeviceInfo();

				unique_ptr<ImageErrorKernel<T>> kernel(
					new ImageErrorKernel<T>(inForwardPropData.Width, inForwardPropData.Height, inForwardPropData.Units,
					inForwardPropMem.WidthOffset, inForwardPropMem.HeightOffset, inForwardPropMem.UnitOffset, inForwardPropMem.Width, inForwardPropMem.Height));

				auto maximumConstantBufferSize = deviceInfo.MaxConstantBufferSize();
				auto inputTargetBytes = sizeof(T) * inForwardPropMem.TotalMemory();
				if (maximumConstantBufferSize > inputTargetBytes) {
					kernel->SetConstantInput(true);
					maximumConstantBufferSize -= inputTargetBytes;
				}

				if (maximumConstantBufferSize > inputTargetBytes) {
					kernel->SetConstantTarget(true);
					maximumConstantBufferSize -= inputTargetBytes;
				}

				kernel->SetUseRelaxedMath(config.UseRelaxedMath());
				kernel->SetComputationPrecision(config.ComputationPrecision());
				kernel->SetErrorFunction(config.ErrorFunction());
				kernel->InitializeCompilerOptions();
				vector<OCLDevice*> oneDeviceVector;
				oneDeviceVector.push_back(device);
				this->context->AddProgramFromSource(kernel.get(), oneDeviceVector);
				this->context->AddKernel(kernel.get());
				deviceAndImageErrorKernels.insert(make_pair(device, move(kernel)));
			}
		}

		template<class T>
		void StandardOutputLayer<T>::InitializeImageOutputKernel()
		{
			auto inForwardPropMem = this->InForwardPropMemoryDescriptions()[0];
			auto inForwardPropData = this->InForwardPropDataDescriptions()[0];
			auto outBackPropMem = this->OutBackPropMemoryDescriptions()[0];
			//In the current implementation, I need to make sure the target (inBackProp)
			//and the (inForwardProp) are the same

			vector<OCLDevice*> devices = this->context->GetDevices();
			for (auto device : devices) {
				auto deviceInfo = device->DeviceInfo();

				unique_ptr<ImageOutputKernel<T>> kernel(
					new ImageOutputKernel<T>(inForwardPropData.Width, inForwardPropData.Height, inForwardPropData.Units,
					inForwardPropMem.WidthOffset, inForwardPropMem.HeightOffset, outBackPropMem.WidthOffset,
					outBackPropMem.HeightOffset, inForwardPropMem.UnitOffset, outBackPropMem.UnitOffset, inForwardPropMem.Width, outBackPropMem.Width,
					outBackPropMem.Height, inForwardPropMem.Height));

				auto maximumConstantBufferSize = deviceInfo.MaxConstantBufferSize();
				auto inputTargetBytes = sizeof(T) * inForwardPropMem.TotalMemory();
				if (maximumConstantBufferSize > inputTargetBytes) {
					kernel->SetConstantInput(true);
					maximumConstantBufferSize -= inputTargetBytes;
				}

				if (maximumConstantBufferSize > inputTargetBytes) {
					kernel->SetConstantTarget(true);
					maximumConstantBufferSize -= inputTargetBytes;
				}

				kernel->SetUseRelaxedMath(config.UseRelaxedMath());
				kernel->SetComputationPrecision(config.ComputationPrecision());
				kernel->SetActivationFunction(this->BackPropActivationFunction());
				kernel->SetErrorFunction(config.ErrorFunction());
				kernel->InitializeCompilerOptions();
				vector<OCLDevice*> oneDeviceVector;
				oneDeviceVector.push_back(device);
				this->context->AddProgramFromSource(kernel.get(), oneDeviceVector);
				this->context->AddKernel(kernel.get());
				deviceAndImageOutputKernels.insert(make_pair(device, move(kernel)));
			}
		}

		template<class T>
		void StandardOutputLayer<T>::InitializeOutputKernel() {

			auto inForwardPropMem = this->InForwardPropMemoryDescriptions()[0];
			auto outBackPropMem = this->OutBackPropMemoryDescriptions()[0];
			//In the current implementation, I need to make sure the target (inBackProp)
			//and the (inForwardProp) are the same

			vector<OCLDevice*> devices = this->context->GetDevices();
			for (auto device : devices) {
				auto deviceInfo = device->DeviceInfo();

				unique_ptr<OutputKernel<T>> kernel(
					new OutputKernel<T>(inputDescription.Units,
					inForwardPropMem.UnitOffset,
					outBackPropMem.UnitOffset));

				auto maximumConstantBufferSize = deviceInfo.MaxConstantBufferSize();
				auto inputTargetBytes = sizeof(T) * inputDescription.Units;
				if (maximumConstantBufferSize > inputTargetBytes) {
					kernel->SetConstantInput(true);
					maximumConstantBufferSize -= inputTargetBytes;
				}
				if (maximumConstantBufferSize > inputTargetBytes) {
					kernel->SetConstantTarget(true);
					maximumConstantBufferSize -= inputTargetBytes;
				}

				kernel->SetUseRelaxedMath(config.UseRelaxedMath());
				kernel->SetComputationPrecision(config.ComputationPrecision());
				kernel->SetActivationFunction(this->BackPropActivationFunction());
				kernel->SetErrorFunction(config.ErrorFunction());
				kernel->InitializeCompilerOptions();
				vector<OCLDevice*> oneDeviceVector;
				oneDeviceVector.push_back(device);
				this->context->AddProgramFromSource(kernel.get(), oneDeviceVector);
				this->context->AddKernel(kernel.get());
				deviceAndOutputKernels.insert(make_pair(device, move(kernel)));
			}
		}

		template<class T>
		T StandardOutputLayer<T>::CalculateError(OCLDevice* device, int queueIndex,
			OCLMemory* previousInput, OCLMemory* target) 
		{

			T result;
			if (useImage)
			{
				auto& kernel = deviceAndImageErrorKernels[device];
				kernel->SetInput(previousInput);
				kernel->SetTarget(target);
				auto errorMemory = context->CreateMemory(CL_MEM_WRITE_ONLY, sizeof(T));
				kernel->SetError(errorMemory.get());
				device->ExecuteTask(kernel.get(), queueIndex, true);
				device->ReadMemory(errorMemory.get(), errorMemory->ByteSize(), &result,
					queueIndex, true);
			}
			else
			{
				auto& kernel = deviceAndErrorKernels[device];
				kernel->SetInput(previousInput);
				kernel->SetTarget(target);
				auto errorMemory = context->CreateMemory(CL_MEM_WRITE_ONLY, sizeof(T));
				kernel->SetError(errorMemory.get());
				device->ExecuteTask(kernel.get(), queueIndex, true);
				device->ReadMemory(errorMemory.get(), errorMemory->ByteSize(), &result,
					queueIndex, true);
			}

			return result;
		}

		template<class T>
		void StandardOutputLayer<T>::EnqueueBackPropagation(OCLDevice* device,
			int queueIndex, OCLMemory* previousInput, OCLMemory* target,
			OCLMemory* deltaOutput, bool blocking) 
		{
			if (useImage)
			{
				auto& kernel = deviceAndImageOutputKernels[device];
				kernel->SetInput(previousInput);
				kernel->SetTarget(target);
				kernel->SetOutput(deltaOutput);
				device->ExecuteKernel(kernel.get(), queueIndex, blocking);
			}
			else
			{
				auto& kernel = deviceAndOutputKernels[device];
				kernel->SetInput(previousInput);
				kernel->SetTarget(target);
				kernel->SetOutput(deltaOutput);
				device->ExecuteKernel(kernel.get(), queueIndex, blocking);
			}
		}

		template class StandardOutputLayer<cl_float> ;
		template class StandardOutputLayer<cl_double> ;

	} /* namespace MachineLearning */
} /* namespace Matuna */
