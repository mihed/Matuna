/*
* PerceptronLayer.cpp
*
*  Created on: May 5, 2015
*      Author: Mikael
*/

#include "PerceptronLayer.h"
#include "CNN/InterlockHelper.h"
#include <stdexcept>
#include <type_traits>
#include <random>

namespace ATML
{
	namespace MachineLearning
	{

		template<class T>
		PerceptronLayer<T>::PerceptronLayer(shared_ptr<OpenCLContext> context,
			const vector<LayerDataDescription>& inputLayerDescriptions,
			ATMLActivationFunction backPropActivation,
			const PerceptronLayerConfig* config) :
		OpenCLForwardBackPropLayer<T>(context, inputLayerDescriptions,
			backPropActivation, config), config(*config)
		{

			if (inputLayerDescriptions.size() == 0)
				throw invalid_argument(
				"There's no input data descriptions for the perceptron layer.");

			if (config->ConnectionType() != ATMLFullConnection)
				throw runtime_error("Not implemented exception");

			//In a perceptron layer, we cannot have multiple input descriptions for the same network
			//since it will correspond to a different weight matrix.
			if (inputLayerDescriptions.size() > 1)
			{
				auto count = inputLayerDescriptions.size();
				for (int i = 1; i < count; i++)
					if (!InterlockHelper::DataEquals(inputLayerDescriptions[i - 1],
						inputLayerDescriptions[i]))
						throw invalid_argument(
						"We cannot have multiple different input descriptions for a perceptron layer");
			}

			for (auto& layerDescription : inputLayerDescriptions)
			{
				LayerMemoryDescription inForwardMemProp;
				inForwardMemProp.Height = layerDescription.Height;
				inForwardMemProp.Width = layerDescription.Width;
				inForwardMemProp.Units = layerDescription.Units;
				inForwardMemProp.HeightOffset = 0;
				inForwardMemProp.UnitOffset = 0;
				inForwardMemProp.WidthOffset = 0;
				this->inForwardPropMemoryProposals.push_back(inForwardMemProp);
				this->outBackPropMemoryProposals.push_back(inForwardMemProp);

				LayerDataDescription outForwardDataDesc;
				outForwardDataDesc.Height = 1;
				outForwardDataDesc.Width = 1;
				outForwardDataDesc.Units = config->Units();
				this->outForwardPropDataDescriptions.push_back(outForwardDataDesc);

				this->inBackPropDataDescriptions = this->outForwardPropDataDescriptions;

				LayerMemoryDescription outForwardMemProp;
				outForwardMemProp.Height = 1;
				outForwardMemProp.Width = 1;
				outForwardMemProp.Units = config->Units();
				outForwardMemProp.HeightOffset = 0;
				outForwardMemProp.UnitOffset = 0;
				outForwardMemProp.WidthOffset = 0;
				this->outForwardPropMemoryProposals.push_back(outForwardMemProp);

				this->inBackPropMemoryProposals.push_back(outForwardMemProp);
			}

			auto inputDataDescriptions = this->InForwardPropDataDescriptions();
			inputDescription = inputDataDescriptions[0];

			scalarCache = nullptr;
			useImage = false;
		}

		template<class T>
		PerceptronLayer<T>::~PerceptronLayer()
		{
			for (auto& deviceAndKernel : deviceAndForwardKernels)
			{
				auto& kernelProgram = deviceAndKernel.second;
				this->context->RemoveKernel(kernelProgram.get());
				this->context->RemoveProgram(kernelProgram.get());
			}

			for (auto& deviceAndKernel : deviceAndBackKernels)
			{
				auto& kernelProgram = deviceAndKernel.second;
				this->context->RemoveKernel(kernelProgram.get());
				this->context->RemoveProgram(kernelProgram.get());
			}

			for (auto& deviceAndKernel : deviceAndGradientKernels)
			{
				auto& kernelProgram = deviceAndKernel.second;
				this->context->RemoveKernel(kernelProgram.get());
				this->context->RemoveProgram(kernelProgram.get());
			}

			for (auto& deviceAndKernel : deviceAndDivideByScalarKernels)
			{
				auto& kernelProgram = deviceAndKernel.second;
				this->context->RemoveKernel(kernelProgram.get());
				this->context->RemoveProgram(kernelProgram.get());
			}

			for (auto& deviceAndKernel : deviceAndSimpleSumKernels)
			{
				auto& kernelProgram = deviceAndKernel.second;
				this->context->RemoveKernel(kernelProgram.get());
				this->context->RemoveProgram(kernelProgram.get());
			}

			for (auto& deviceAndKernel : deviceAndImageBackKernels)
			{
				auto& kernelProgram = deviceAndKernel.second;
				this->context->RemoveKernel(kernelProgram.get());
				this->context->RemoveProgram(kernelProgram.get());
			}

			for (auto& deviceAndKernel : deviceAndImageGradientKernels)
			{
				auto& kernelProgram = deviceAndKernel.second;
				this->context->RemoveKernel(kernelProgram.get());
				this->context->RemoveProgram(kernelProgram.get());
			}
		}

		template<class T>
		PerceptronLayerConfig PerceptronLayer<T>::GetConfig() const
		{
			return config;
		}

		template<class T>
		Matrix<T> PerceptronLayer<T>::GetWeights()
		{
			OpenCLDevice* device = this->context->GetDevices()[0];
			Matrix<T> result(config.Units(), inputDescription.TotalUnits());
			device->ReadMemory(weights.get(), weights->ByteSize(), result.Data, 0,
				true);

			return result;
		}

		template<class T>
		Matrix<T> PerceptronLayer<T>::GetBias()
		{
			OpenCLDevice* device = this->context->GetDevices()[0];
			Matrix<T> result(config.Units(), 1);
			device->ReadMemory(biases.get(), biases->ByteSize(), result.Data, 0, true);

			return result;
		}

		template<class T>
		void PerceptronLayer<T>::InterlockFinalized()
		{
			auto inputMemoryDescriptions = this->InForwardPropMemoryDescriptions();
			auto& firstMemory = inputMemoryDescriptions[0];

			InitializeParameters();

			//IF the memory descriptions doesn't contain any padding or offsets, we may use the standard forward prop kernel.
			//FIXME: A normal perceptron can handle offset in the unit direction, we don't need to fall back onto the image perceptron.
			if (firstMemory.HeightOffset == 0 && firstMemory.UnitOffset == 0
				&& firstMemory.WidthOffset == 0 && firstMemory.Width == 1
				&& firstMemory.Height == 1
				&& firstMemory.Units == inputDescription.Units)
			{
				InitializeNormalForwardPerceptron();
				InitializeNormalBackPerceptron();
				InitializeNormalGradientKernel();
				useImage = false;
			}
			else
			{
				InitializeImageForwardPerceptron();
				InitializeImageBackPerceptron();
				InitializeImageGradientKernel();
				useImage = true;
			}
		}

		template<class T>
		void PerceptronLayer<T>::InitializeNormalGradientKernel()
		{
			//The memory maps to the data description in this case
			auto outputDataDescriptions = this->outForwardPropDataDescriptions;
			auto& firstOutputData = outputDataDescriptions[0];

			vector<OpenCLDevice*> devices = this->context->GetDevices();

			for (auto device : devices)
			{
				auto deviceInfo = device->DeviceInfo();

				//Make sure the type we want to execute is supported on the device.
				if (is_same<cl_double, T>::value)
				{
					if (deviceInfo.PreferredDoubleVectorWidth() == 0)
						throw invalid_argument(
						"The template argument is not supported on the chosen devices");
				}
				else if (is_same<cl_float, T>::value)
				{
					if (deviceInfo.PreferredFloatVectorWidth() == 0)
						throw invalid_argument(
						"The template argument is not supported on the chosen devices");
				}
				else
					throw runtime_error(
					"The template argument does not match the supported arguments");

				unique_ptr<GradientPerceptronKernel<T>> kernel(
					new GradientPerceptronKernel<T>(inputDescription.TotalUnits(),
					firstOutputData.TotalUnits()));

				//Now, let us query the device if we have enough memory to use constant weights / inputs / biases etc...
				auto maximumConstantBufferSize = deviceInfo.MaxConstantBufferSize();
				auto byteSize = firstOutputData.TotalUnits() * sizeof(T);
				if (maximumConstantBufferSize > byteSize)
				{
					kernel->SetConstantInputDelta(true);
					maximumConstantBufferSize -= byteSize;
				}

				byteSize = inputDescription.TotalUnits() * sizeof(T);
				if (maximumConstantBufferSize > byteSize)
				{
					kernel->SetConstantInput(true);
					maximumConstantBufferSize -= byteSize;
				}

				kernel->SetUseRelaxedMath(config.UseRelaxedMath());
				kernel->InitializeCompilerOptions();
				vector<OpenCLDevice*> oneDeviceVector;
				oneDeviceVector.push_back(device);
				this->context->AddProgramFromSource(kernel.get(), oneDeviceVector);
				this->context->AddKernel(kernel.get());
				deviceAndGradientKernels.insert(make_pair(device, move(kernel)));
			}
		}

		template<class T>
		void PerceptronLayer<T>::InitializeImageGradientKernel()
		{
			//The memory maps to the data description in this case
			LayerDataDescription firstOutputData =
				this->outForwardPropDataDescriptions[0];
			LayerMemoryDescription firstInputMemDesc =
				this->InForwardPropMemoryDescriptions()[0];

			vector<OpenCLDevice*> devices = this->context->GetDevices();

			for (auto device : devices)
			{
				auto deviceInfo = device->DeviceInfo();

				//Make sure the type we want to execute is supported on the device.
				if (is_same<cl_double, T>::value)
				{
					if (deviceInfo.PreferredDoubleVectorWidth() == 0)
						throw invalid_argument(
						"The template argument is not supported on the chosen devices");
				}
				else if (is_same<cl_float, T>::value)
				{
					if (deviceInfo.PreferredFloatVectorWidth() == 0)
						throw invalid_argument(
						"The template argument is not supported on the chosen devices");
				}
				else
					throw runtime_error(
					"The template argument does not match the supported arguments");

				unique_ptr<ImageGradientPerceptronKernel<T>> kernel(
					new ImageGradientPerceptronKernel<T>(
					inputDescription.TotalUnits(),
					firstOutputData.TotalUnits(), inputDescription.Width,
					inputDescription.Height, firstInputMemDesc.WidthOffset,
					firstInputMemDesc.HeightOffset,
					firstInputMemDesc.UnitOffset, firstInputMemDesc.Width,
					firstInputMemDesc.Height, 0,
					inputDescription.TotalUnits()));

				//Now, let us query the device if we have enough memory to use constant weights / inputs / biases etc...
				auto maximumConstantBufferSize = deviceInfo.MaxConstantBufferSize();
				auto byteSize = firstOutputData.TotalUnits() * sizeof(T);
				if (maximumConstantBufferSize > byteSize)
				{
					kernel->SetConstantInputDelta(true);
					maximumConstantBufferSize -= byteSize;
				}

				byteSize = firstInputMemDesc.TotalMemory() * sizeof(T);
				if (maximumConstantBufferSize > byteSize)
				{
					kernel->SetConstantInput(true);
					maximumConstantBufferSize -= byteSize;
				}

				kernel->SetUseRelaxedMath(config.UseRelaxedMath());
				kernel->InitializeCompilerOptions();
				vector<OpenCLDevice*> oneDeviceVector;
				oneDeviceVector.push_back(device);
				this->context->AddProgramFromSource(kernel.get(), oneDeviceVector);
				this->context->AddKernel(kernel.get());
				deviceAndImageGradientKernels.insert(make_pair(device, move(kernel)));
			}

		}

		template<class T>
		void PerceptronLayer<T>::InitializeNormalBackPerceptron()
		{
			//The memory maps to the data description in this case
			auto outputDataDescriptions = this->outForwardPropDataDescriptions;
			auto& firstOutputData = outputDataDescriptions[0];

			int biasCount = firstOutputData.TotalUnits();

			vector<OpenCLDevice*> devices = this->context->GetDevices();

			for (auto device : devices)
			{
				auto deviceInfo = device->DeviceInfo();

				//Make sure the type we want to execute is supported on the device.
				if (is_same<cl_double, T>::value)
				{
					if (deviceInfo.PreferredDoubleVectorWidth() == 0)
						throw invalid_argument(
						"The template argument is not supported on the chosen devices");
				}
				else if (is_same<cl_float, T>::value)
				{
					if (deviceInfo.PreferredFloatVectorWidth() == 0)
						throw invalid_argument(
						"The template argument is not supported on the chosen devices");
				}
				else
					throw runtime_error(
					"The template argument does not match the supported arguments");

				unique_ptr<BackPerceptronKernel<T>> kernel(
					new BackPerceptronKernel<T>(firstOutputData.TotalUnits(),
					inputDescription.TotalUnits(), 0, 0, 0));

				//Now, let us query the device if we have enough memory to use constant weights / inputs / biases etc...
				auto maximumConstantBufferSize = deviceInfo.MaxConstantBufferSize();
				if (maximumConstantBufferSize > weights->ByteSize())
				{
					kernel->SetUseConstantWeights(true);
					maximumConstantBufferSize -= weights->ByteSize();
				}

				auto byteSize = firstOutputData.TotalUnits() * sizeof(T);
				if (maximumConstantBufferSize > byteSize)
				{
					kernel->SetUseConstantDeltaInput(true);
					maximumConstantBufferSize -= byteSize;
				}

				byteSize = inputDescription.TotalUnits() * sizeof(T);
				if (maximumConstantBufferSize > byteSize)
				{
					kernel->SetUseConstantInput(true);
					maximumConstantBufferSize -= byteSize;
				}

				kernel->SetUseRelaxedMath(config.UseRelaxedMath());
				kernel->SetActivationFunction(this->BackPropActivationFunction());
				kernel->SetWeights(weights.get());
				kernel->InitializeCompilerOptions();
				vector<OpenCLDevice*> oneDeviceVector;
				oneDeviceVector.push_back(device);
				this->context->AddProgramFromSource(kernel.get(), oneDeviceVector);
				this->context->AddKernel(kernel.get());
				kernel->InitializeArguments();
				deviceAndBackKernels.insert(make_pair(device, move(kernel)));
			}
		}

		template<class T>
		void PerceptronLayer<T>::InitializeImageBackPerceptron()
		{

			LayerDataDescription outForwardDataDesc =
				this->outForwardPropDataDescriptions[0];
			LayerDataDescription inForwardDataDesc =
				this->InForwardPropDataDescriptions()[0];
			LayerMemoryDescription inBackMemDesc =
				this->InBackPropMemoryDescriptions()[0];
			LayerMemoryDescription inForwardMemDesc =
				this->InForwardPropMemoryDescriptions()[0];
			LayerMemoryDescription outBackMemDesc =
				this->OutBackPropMemoryDescriptions()[0];

			vector<OpenCLDevice*> devices = this->context->GetDevices();

			for (auto device : devices)
			{
				auto deviceInfo = device->DeviceInfo();

				//Make sure the type we want to execute is supported on the device.
				if (is_same<cl_double, T>::value)
				{
					if (deviceInfo.PreferredDoubleVectorWidth() == 0)
						throw invalid_argument(
						"The template argument is not supported on the chosen devices");
				}
				else if (is_same<cl_float, T>::value)
				{
					if (deviceInfo.PreferredFloatVectorWidth() == 0)
						throw invalid_argument(
						"The template argument is not supported on the chosen devices");
				}
				else
					throw runtime_error(
					"The template argument does not match the supported arguments");

				unique_ptr<ImageBackPerceptronKernel<T>> kernel(
					new ImageBackPerceptronKernel<T>(inForwardDataDesc.Width,
					inForwardDataDesc.Height, inForwardDataDesc.Units,
					outBackMemDesc.WidthOffset, outBackMemDesc.HeightOffset,
					outBackMemDesc.UnitOffset, inForwardMemDesc.WidthOffset,
					inForwardMemDesc.HeightOffset,
					inForwardMemDesc.UnitOffset, outBackMemDesc.Width,
					outBackMemDesc.Height, inForwardMemDesc.Width,
					inForwardMemDesc.Height, inForwardDataDesc.TotalUnits(),
					inBackMemDesc.UnitOffset, outForwardDataDesc.Units));

				//Now, let us query the device if we have enough memory to use constant weights / inputs / biases etc...
				auto maximumConstantBufferSize = deviceInfo.MaxConstantBufferSize();
				if (maximumConstantBufferSize > weights->ByteSize())
				{
					kernel->SetUseConstantWeights(true);
					maximumConstantBufferSize -= weights->ByteSize();
				}

				auto byteSize = inBackMemDesc.TotalMemory() * sizeof(T);
				if (maximumConstantBufferSize > byteSize)
				{
					kernel->SetUseConstantDeltaInput(true);
					maximumConstantBufferSize -= byteSize;
				}

				byteSize = inForwardMemDesc.TotalMemory() * sizeof(T);
				if (maximumConstantBufferSize > byteSize)
				{
					kernel->SetUseConstantInput(true);
					maximumConstantBufferSize -= byteSize;
				}

				kernel->SetUseRelaxedMath(config.UseRelaxedMath());
				kernel->SetActivationFunction(this->BackPropActivationFunction());
				kernel->InitializeCompilerOptions();

				vector<OpenCLDevice*> oneDeviceVector;
				oneDeviceVector.push_back(device);

				this->context->AddProgramFromSource(kernel.get(), oneDeviceVector);
				this->context->AddKernel(kernel.get());

				kernel->SetWeights(weights.get());
				deviceAndImageBackKernels.insert(make_pair(device, move(kernel)));
			}
		}

		template<class T>
		void PerceptronLayer<T>::InitializeNormalForwardPerceptron()
		{
			auto outputDataDescriptions = this->outForwardPropDataDescriptions;
			auto& firstOutputData = outputDataDescriptions[0];

			int biasCount = firstOutputData.TotalUnits();

			vector<OpenCLDevice*> devices = this->context->GetDevices();

			for (auto device : devices)
			{

				vector<OpenCLDevice*> oneDeviceVector;
				oneDeviceVector.push_back(device);
				auto deviceInfo = device->DeviceInfo();

				//Make sure the type we want to execute is supported on the device.
				if (is_same<cl_double, T>::value)
				{
					if (deviceInfo.PreferredDoubleVectorWidth() == 0)
						throw invalid_argument(
						"The template argument is not supported on the chosen devices");
				}
				else if (is_same<cl_float, T>::value)
				{
					if (deviceInfo.PreferredFloatVectorWidth() == 0)
						throw invalid_argument(
						"The template argument is not supported on the chosen devices");
				}
				else
					throw runtime_error(
					"The template argument does not match the supported arguments");

				if (config.ActivationFunction() == ATMLSoftMaxActivation)
				{
					unique_ptr<SimpleSumKernel<T>> sumKernel(
						new SimpleSumKernel<T>(biasCount));
					unique_ptr<DivideByScalarKernel<T>> scalarKernel(
						new DivideByScalarKernel<T>(biasCount));
					scalarCache = move(
						this->context->CreateMemory(CL_MEM_READ_WRITE, sizeof(T)));
					this->context->AddProgramFromSource(sumKernel.get(),
						oneDeviceVector);
					this->context->AddKernel(sumKernel.get());
					this->context->AddProgramFromSource(scalarKernel.get(),
						oneDeviceVector);
					this->context->AddKernel(scalarKernel.get());

					deviceAndDivideByScalarKernels.insert(
						make_pair(device, move(scalarKernel)));
					deviceAndSimpleSumKernels.insert(
						make_pair(device, move(sumKernel)));
				}

				unique_ptr<ForwardPerceptronKernel<T>> kernel(
					new ForwardPerceptronKernel<T>(inputDescription.TotalUnits(),
					firstOutputData.TotalUnits()));

				//Now, let us query the device if we have enough memory to use constant weights / inputs / biases etc...
				auto maximumConstantBufferSize = deviceInfo.MaxConstantBufferSize();
				auto biasBytes = sizeof(T) * biasCount;
				if (maximumConstantBufferSize > weights->ByteSize())
				{
					kernel->SetUseConstantWeights(true);
					maximumConstantBufferSize -= weights->ByteSize();
				}
				if (maximumConstantBufferSize > biasBytes)
				{
					kernel->SetUseConstantInput(true);
					maximumConstantBufferSize -= biasBytes;
				}
				if (maximumConstantBufferSize > biasBytes)
				{
					kernel->SetUseConstantBiases(true);
					maximumConstantBufferSize -= biasBytes;
				}

				kernel->SetUseRelaxedMath(config.UseRelaxedMath());
				kernel->SetComputationPrecision(config.ComputationPrecision());
				kernel->SetActivationFunction(config.ActivationFunction());
				kernel->SetWeights(weights.get());
				kernel->SetBiases(biases.get());
				kernel->InitializeCompilerOptions();
				this->context->AddProgramFromSource(kernel.get(), oneDeviceVector);
				this->context->AddKernel(kernel.get());
				kernel->InitializeArguments();
				deviceAndForwardKernels.insert(make_pair(device, move(kernel)));
			}
		}

		template<class T>
		void PerceptronLayer<T>::InitializeParameters()
		{
			auto outputDataDescriptions = this->outForwardPropDataDescriptions;
			auto& firstOutputData = outputDataDescriptions[0];

			int weightCount = inputDescription.TotalUnits()
				* firstOutputData.TotalUnits();

			int biasCount = firstOutputData.TotalUnits();

			//TODO: Here's optimization to be made in case the network is read-only and not trainable.
			weights = move(
				this->context->CreateMemory(CL_MEM_READ_WRITE,
				sizeof(T) * weightCount));

			biases = move(
				this->context->CreateMemory(CL_MEM_READ_WRITE,
				sizeof(T) * biasCount));

			random_device tempDevice;
			mt19937 mt(tempDevice());

			//TODO: The initial weight values could be something to tweak
			uniform_real_distribution<T> uniformDistribution(-0.1, 0.1);

			vector<T> initialWeightValues;
			initialWeightValues.resize(weightCount);
			for (int i = 0; i < weightCount; i++)
				initialWeightValues[i] = uniformDistribution(mt);

			vector<T> initialBiasValues;
			initialBiasValues.resize(biasCount);
			for (int i = 0; i < biasCount; i++)
				initialBiasValues[i] = uniformDistribution(mt);

			//Since this is initialization, we don't really care about which device and device queue we are using
			OpenCLDevice* device = this->context->GetDevices()[0];
			device->WriteMemory(weights.get(), sizeof(T) * initialWeightValues.size(),
				initialWeightValues.data(), 0, false);
			device->WriteMemory(biases.get(), sizeof(T) * initialBiasValues.size(),
				initialBiasValues.data(), 0, false);
			device->WaitForDeviceQueue(0);
		}

		template<class T>
		void PerceptronLayer<T>::InitializeImageForwardPerceptron()
		{
			LayerDataDescription firstOutputData =
				this->outForwardPropDataDescriptions[0];
			LayerDataDescription firstInputData =
				this->InForwardPropDataDescriptions()[0];
			LayerMemoryDescription firstInputMemory =
				this->InForwardPropMemoryDescriptions()[0];

			int biasCount = firstOutputData.TotalUnits();

			vector<OpenCLDevice*> devices = this->context->GetDevices();

			for (auto device : devices)
			{

				vector<OpenCLDevice*> oneDeviceVector;
				oneDeviceVector.push_back(device);
				auto deviceInfo = device->DeviceInfo();

				//Make sure the type we want to execute is supported on the device.
				if (is_same<cl_double, T>::value)
				{
					if (deviceInfo.PreferredDoubleVectorWidth() == 0)
						throw invalid_argument(
						"The template argument is not supported on the chosen devices");
				}
				else if (is_same<cl_float, T>::value)
				{
					if (deviceInfo.PreferredFloatVectorWidth() == 0)
						throw invalid_argument(
						"The template argument is not supported on the chosen devices");
				}
				else
					throw runtime_error(
					"The template argument does not match the supported arguments");

				if (config.ActivationFunction() == ATMLSoftMaxActivation)
				{
					unique_ptr<SimpleSumKernel<T>> sumKernel(
						new SimpleSumKernel<T>(biasCount));
					unique_ptr<DivideByScalarKernel<T>> scalarKernel(
						new DivideByScalarKernel<T>(biasCount));
					scalarCache = move(
						this->context->CreateMemory(CL_MEM_READ_WRITE, sizeof(T)));
					this->context->AddProgramFromSource(sumKernel.get(),
						oneDeviceVector);
					this->context->AddKernel(sumKernel.get());
					this->context->AddProgramFromSource(scalarKernel.get(),
						oneDeviceVector);
					this->context->AddKernel(scalarKernel.get());

					deviceAndDivideByScalarKernels.insert(
						make_pair(device, move(scalarKernel)));
					deviceAndSimpleSumKernels.insert(
						make_pair(device, move(sumKernel)));
				}

				unique_ptr<ImageForwardPerceptronKernel<T>> kernel(
					new ImageForwardPerceptronKernel<T>(biasCount,
					firstInputData.Units, firstInputData.Width,
					firstInputData.Height, firstInputMemory.WidthOffset,
					firstInputMemory.HeightOffset,
					firstInputMemory.UnitOffset, firstInputData.Width,
					firstInputData.Height, firstInputData.TotalUnits(), 0));

				//Now, let us query the device if we have enough memory to use constant weights / inputs / biases etc...
				auto maximumConstantBufferSize = deviceInfo.MaxConstantBufferSize();
				auto biasBytes = sizeof(T) * biasCount;
				auto inputBytes = sizeof(T) * firstInputMemory.TotalMemory();
				if (maximumConstantBufferSize > inputBytes)
				{
					kernel->SetUseConstantInput(true);
					maximumConstantBufferSize -= inputBytes;
				}
				if (maximumConstantBufferSize > weights->ByteSize())
				{
					kernel->SetUseConstantWeights(true);
					maximumConstantBufferSize -= weights->ByteSize();
				}
				if (maximumConstantBufferSize > biasBytes)
				{
					kernel->SetUseConstantBiases(true);
					maximumConstantBufferSize -= biasBytes;
				}

				kernel->SetUseRelaxedMath(config.UseRelaxedMath());
				kernel->SetComputationPrecision(config.ComputationPrecision());
				kernel->SetActivationFunction(config.ActivationFunction());
				kernel->InitializeCompilerOptions();

				this->context->AddProgramFromSource(kernel.get(), oneDeviceVector);
				this->context->AddKernel(kernel.get());

				kernel->SetWeights(weights.get());
				kernel->SetBiases(biases.get());

				deviceAndImageForwardKernels.insert(make_pair(device, move(kernel)));
			}
		}

		template<class T>
		void PerceptronLayer<T>::EnqueueForwardPropagation(OpenCLDevice* device,
			int queueIndex, OpenCLMemory* previousInput, OpenCLMemory* output,
			bool blocking)
		{
			if (useImage)
			{
				auto& kernel = deviceAndImageForwardKernels[device];
				kernel->SetInput(previousInput);
				kernel->SetOutput(output);
				if (config.ActivationFunction() == ATMLSoftMaxActivation)
				{
					device->ExecuteKernel(kernel.get(), queueIndex, false);
					auto& sumKernel = deviceAndSimpleSumKernels[device];
					auto& scalarKernel = deviceAndDivideByScalarKernels[device];
					sumKernel->SetInput(output);
					sumKernel->SetOutput(scalarCache.get());
					device->ExecuteTask(sumKernel.get(), queueIndex, false);
					scalarKernel->SetInputOutput(output);
					scalarKernel->SetScalar(scalarCache.get());
					device->ExecuteKernel(scalarKernel.get(), queueIndex, blocking);
				}
				else
					device->ExecuteKernel(kernel.get(), queueIndex, blocking);
			}
			else
			{
				auto& kernel = deviceAndForwardKernels[device];
				kernel->SetInput(previousInput);
				kernel->SetOutput(output);
				if (config.ActivationFunction() == ATMLSoftMaxActivation)
				{
					device->ExecuteKernel(kernel.get(), queueIndex, false);
					auto& sumKernel = deviceAndSimpleSumKernels[device];
					auto& scalarKernel = deviceAndDivideByScalarKernels[device];
					sumKernel->SetInput(output);
					sumKernel->SetOutput(scalarCache.get());
					device->ExecuteTask(sumKernel.get(), queueIndex, false);
					scalarKernel->SetInputOutput(output);
					scalarKernel->SetScalar(scalarCache.get());
					device->ExecuteKernel(scalarKernel.get(), queueIndex, blocking);
				}
				else
					device->ExecuteKernel(kernel.get(), queueIndex, blocking);
			}
		}

		template<class T>
		void PerceptronLayer<T>::EnqueueBackPropagation(OpenCLDevice* device,
			int queueIndex, OpenCLMemory* previousInput, OpenCLMemory* delta,
			OpenCLMemory* deltaOutput, bool blocking)
		{
			if (useImage)
			{
				auto& kernel = deviceAndImageBackKernels[device];
				kernel->SetInput(previousInput);
				kernel->SetDeltaInput(delta);
				kernel->SetOutput(deltaOutput);
				device->ExecuteKernel(kernel.get(), queueIndex, blocking);
			}
			else
			{
				auto& kernel = deviceAndBackKernels[device];
				kernel->SetInput(previousInput);
				kernel->SetDeltaInput(delta);
				kernel->SetOutput(deltaOutput);
				device->ExecuteKernel(kernel.get(), queueIndex, blocking);
			}
		}

		template<class T>
		void PerceptronLayer<T>::EnqueueCalculateGradient(OpenCLDevice* device,
			int queueIndex, OpenCLMemory* previousInput, OpenCLMemory* delta,
			OpenCLMemory* gradient, bool blocking)
		{
			if (useImage)
			{
				auto& kernel = deviceAndImageGradientKernels[device];
				kernel->SetInput(previousInput);
				kernel->SetInputDelta(delta);
				kernel->SetGradient(gradient);
				device->ExecuteKernel(kernel.get(), queueIndex, blocking);
			}
			else
			{
				auto& kernel = deviceAndGradientKernels[device];
				kernel->SetInput(previousInput);
				kernel->SetInputDelta(delta);
				kernel->SetGradient(gradient);
				device->ExecuteKernel(kernel.get(), queueIndex, blocking);
			}

			//Since we don't need to calculate anything for the bias gradient, we simply use copy buffer.
			device->CopyCLMemory(delta, gradient, 0,
				config.Units() * inputDescription.TotalUnits() * sizeof(T),
				config.Units() * sizeof(T), queueIndex, blocking);
		}

		template<class T>
		void PerceptronLayer<T>::EnqueueCalculateGradient(OpenCLDevice* device, int queueIndex,
			OpenCLMemory* previousInput, OpenCLMemory* delta, vector<OpenCLMemory*> gradient, bool blocking)
		{

			if (gradient.size() != 2)
				throw invalid_argument("The gradient size is not valid");

			if (gradient[0]->ByteSize() / sizeof(T) != (inputDescription.TotalUnits() * config.Units()))
				throw invalid_argument("The first gradient does not contain the correct amount of memory");

			if (gradient[1]->ByteSize() / sizeof(T) != (config.Units()))
				throw invalid_argument("The second gradient does not contain the correct amount of memory");

			if (useImage)
			{
				auto& kernel = deviceAndImageGradientKernels[device];
				kernel->SetInput(previousInput);
				kernel->SetInputDelta(delta);
				kernel->SetGradient(gradient[0]);
				device->ExecuteKernel(kernel.get(), queueIndex, blocking);
			}
			else
			{
				auto& kernel = deviceAndGradientKernels[device];
				kernel->SetInput(previousInput);
				kernel->SetInputDelta(delta);
				kernel->SetGradient(gradient[0]);
				device->ExecuteKernel(kernel.get(), queueIndex, blocking);
			}

			//Since we don't need to calculate anything for the bias gradient, we simply use copy buffer.
			device->CopyCLMemory(delta, gradient[1], 0, 0,
				config.Units() * sizeof(T), queueIndex, blocking);
		}

		template<class T>
		vector<size_t> PerceptronLayer<T>::GetMultipleParameterCount()
		{
			vector<size_t> result;
			result.push_back(inputDescription.TotalUnits() * config.Units());
			result.push_back(config.Units());
			return result;
		}

		template<class T>
		vector<OpenCLMemory*> PerceptronLayer<T>::GetParameters()
		{
			vector<OpenCLMemory*> result;
			result.push_back(weights.get());
			result.push_back(biases.get());
			return result;
		}

		template<class T>
		void PerceptronLayer<T>::GetParameters(T* parameters, OpenCLDevice* device,
			int queueIndex, bool blocking)
		{
			device->ReadMemory(weights.get(), weights->ByteSize(), parameters,
				queueIndex, blocking);
			auto biasPosition = parameters
				+ config.Units() * inputDescription.TotalUnits();
			device->ReadMemory(biases.get(), biases->ByteSize(), biasPosition,
				queueIndex, blocking);
		}

		template<class T>
		void PerceptronLayer<T>::SetParameters(T* parameters, OpenCLDevice* device,
			int queueIndex, bool blocking)
		{
			device->WriteMemory(weights.get(), weights->ByteSize(), parameters,
				queueIndex, blocking);
			auto biasPosition = parameters
				+ config.Units() * inputDescription.TotalUnits();
			device->WriteMemory(biases.get(), biases->ByteSize(), biasPosition,
				queueIndex, blocking);
		}

		template<class T>
		size_t PerceptronLayer<T>::GetParameterCount()
		{
			return inputDescription.TotalUnits() * config.Units() + config.Units();
		}

		template class PerceptronLayer<cl_float> ;
		template class PerceptronLayer<cl_double> ;

	} /* namespace MachineLearning */
} /* namespace ATML */
