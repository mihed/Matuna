/*
* PerceptronLayer.cpp
*
*  Created on: May 5, 2015
*      Author: Mikael
*/

#include "PerceptronLayer.h"
#include "Matuna.OCLHelper/OCLProgram.h"
#include "Matuna.ConvNet/InterlockHelper.h"
#include "Matuna.Helper/Path.h"
#include "Matuna.Helper/FileHelper.h"
#include <stdexcept>
#include <type_traits>
#include <random>

namespace Matuna
{
	namespace MachineLearning
	{

		template<class T>
		PerceptronLayer<T>::PerceptronLayer(shared_ptr<OCLContext> context,
			const vector<LayerDataDescription>& inputLayerDescriptions,
			MatunaActivationFunction backPropActivation,
			const PerceptronLayerConfig* config) :
		OCLForwardBackPropLayer<T>(context, inputLayerDescriptions,
			backPropActivation, config), config(*config)
		{

			if (inputLayerDescriptions.size() == 0)
				throw invalid_argument(
				"There's no input data descriptions for the perceptron layer.");

			if (config->ConnectionType() != MatunaFullConnection)
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

		}

		template<class T>
		PerceptronLayerConfig PerceptronLayer<T>::GetConfig() const
		{
			return config;
		}

		template<class T>
		Matrix<T> PerceptronLayer<T>::GetWeights()
		{
			OCLDevice* device = this->context->GetDevices()[0];
			Matrix<T> result(config.Units(), inputDescription.TotalUnits());
			device->ReadMemory(weights.get(), weights->ByteSize(), result.Data, 0,
				true);

			return result;
		}

		template<class T>
		Matrix<T> PerceptronLayer<T>::GetBias()
		{
			OCLDevice* device = this->context->GetDevices()[0];
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


			unordered_map<OCLDevice*, unique_ptr<OCLProgram>> programs;
			auto backActivationFunction = this->BackPropActivationFunction();
			auto activationFunction = config.ActivationFunction();
			for (auto device : devices)
			{
				unique_ptr<OCLProgram> program(new OCLProgram());

				program->SetUseRelaxedMath(config.UseRelaxedMath());
				if (config.ComputationPrecision() == MatunaHalfPrecision)
					program->AddDefine("HALF_MATH");
				else if (config.ComputationPrecision() == MatunaNativePrecision)
					program->AddDefine("NATIVE_MATH");

				program->AddIncludePath(OCLProgram::DefaultSourceLocation);

				if (is_same<cl_double, T>::value) 
					program->AddDefine("DOUBLE_PRECISION");

				if (backActivationFunction == MatunaSigmoidActivation)
					program->AddDefine("MATUNA_ACTIVATION_DERIVATIVE_SIGMOID");
				else if (backActivationFunction == MatunaTanhActivation)
					program->AddDefine("MATUNA_ACTIVATION_DERIVATIVE_TANH");
				else if (backActivationFunction == MatunaSoftMaxActivation)
					throw invalid_argument("Softmax is not allowed in other layers than the last one");

				if (activationFunction == MatunaSigmoidActivation)
					program->AddDefine("MATUNA_ACTIVATION_SIGMOID");
				else if (activationFunction == MatunaTanhActivation)
					program->AddDefine("MATUNA_ACTIVATION_TANH");
				else if (activationFunction == MatunaSoftMaxActivation)
					program->AddDefine("MATUNA_ACTIVATION_SOFTMAX");

				program->SetName("PerceptronLayerProgram" + to_string(program->InstanceCount()));
				programs.insert(make_pair(device, move(program)));
			}

			//FIXME: A normal perceptron can handle offset in the unit direction, we don't need to fall back onto the image perceptron.
			if (firstMemory.HeightOffset == 0 && firstMemory.UnitOffset == 0
				&& firstMemory.WidthOffset == 0 && firstMemory.Width == 1
				&& firstMemory.Height == 1
				&& firstMemory.Units == inputDescription.Units)
			{
				InitializeProgram(programs);


				//Attaching and compiling all the programs 
				for (auto device : devices)
				{
					vector<OCLDevice*> oneDeviceVector;
					oneDeviceVector.push_back(device);
					context->AttachProgram(move(programs[device]), oneDeviceVector);

					deviceAndForwardKernels[device]->SetMemoryArg(weights.get(), 2);
					deviceAndForwardKernels[device]->SetMemoryArg(biases.get(), 3);

					deviceAndBackKernels[device]->SetMemoryArg(weights.get(), 3);
				}

				useImage = false;
			}
			else
			{
				InitializeImageProgram(programs);


				//Attaching and compiling all the programs 
				for (auto device : devices)
				{
					vector<OCLDevice*> oneDeviceVector;
					oneDeviceVector.push_back(device);
					context->AttachProgram(move(programs[device]), oneDeviceVector);

					LayerKernel<T>* test = deviceAndImageForwardKernels[device];

					deviceAndImageForwardKernels[device]->SetMemoryArg(weights.get(), 2);
					deviceAndImageForwardKernels[device]->SetMemoryArg(biases.get(), 3);

					deviceAndImageBackKernels[device]->SetMemoryArg(weights.get(), 3);
				}

				useImage = true;
			}

		}

		template<class T>
		void PerceptronLayer<T>::InitializeImageProgram(unordered_map<OCLDevice*, unique_ptr<OCLProgram>>& programs)
		{
			//TODO: FIXME, we have duplicated names. This is just to avoid error when putting into the new form
			LayerDataDescription firstOutputData =
				this->outForwardPropDataDescriptions[0];
			LayerMemoryDescription firstInputMemDesc =
				this->InForwardPropMemoryDescriptions()[0];

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

			string gradientPath = Path::Combine(OCLProgram::DefaultSourceLocation, "ImageGradientPerceptronKernel.cl");
			string backPropPath = Path::Combine(OCLProgram::DefaultSourceLocation, "ImageBackPropPerceptronKernel.cl");
			string forwardPropPath = Path::Combine(OCLProgram::DefaultSourceLocation, "ImageForwardPerceptronKernel.cl");
			string simpleSumPath = Path::Combine(OCLProgram::DefaultSourceLocation, "SimpleSumKernel.cl");
			string divideByScalarPath = Path::Combine(OCLProgram::DefaultSourceLocation, "DivideByScalarKernel.cl");

			vector<OCLDevice*> devices = this->context->GetDevices();

			//Starting with the gradient
			for (auto device : devices)
			{
				auto deviceInfo = device->DeviceInfo();

				unique_ptr<LayerKernel<T>> kernel(new LayerKernel<T>());

				kernel->AddIncludePath(OCLProgram::DefaultSourceLocation);
				kernel->AddSourcePath(gradientPath);
				kernel->SetKernelName("ImageGradientPerceptronKernel");

				//Now, let us query the device if we have enough memory to use constant weights / inputs / biases etc...
				auto maximumConstantBufferSize = deviceInfo.MaxConstantBufferSize();
				auto byteSize = firstOutputData.TotalUnits() * sizeof(T);
				if (maximumConstantBufferSize > byteSize)
				{
					kernel->AddDefine(gradientPath, "CONSTANT_INPUT");
					maximumConstantBufferSize -= byteSize;
				}

				byteSize = firstInputMemDesc.TotalMemory() * sizeof(T);
				if (maximumConstantBufferSize > byteSize)
				{
					kernel->AddDefine(gradientPath, "CONSTANT_INPUT_DELTA");
					maximumConstantBufferSize -= byteSize;
				}

				kernel->AddGlobalSize(inputDescription.TotalUnits());
				kernel->AddGlobalSize(firstOutputData.TotalUnits());

				kernel->AddDefineSubsitute(gradientPath, "INPUT_DATA_WIDTH", to_string(inputDescription.Width));
				kernel->AddDefineSubsitute(gradientPath, "INPUT_UNIT_ELEMENT_COUNT", to_string(inputDescription.Height * inputDescription.Width));
				kernel->AddDefineSubsitute(gradientPath, "INPUT_WIDTH_OFFSET", to_string(firstInputMemDesc.WidthOffset));
				kernel->AddDefineSubsitute(gradientPath, "INPUT_HEIGHT_OFFSET", to_string(firstInputMemDesc.HeightOffset));
				kernel->AddDefineSubsitute(gradientPath, "INPUT_UNIT_OFFSET", to_string(firstInputMemDesc.UnitOffset));
				kernel->AddDefineSubsitute(gradientPath, "INPUT_STRIDE", to_string(firstInputMemDesc.Width));
				kernel->AddDefineSubsitute(gradientPath, "INPUT_UNIT_ELEMENT_COUNT_INC_PADDING", to_string(firstInputMemDesc.Width * firstInputMemDesc.Height));
				kernel->AddDefineSubsitute(gradientPath, "INPUT_DELTA_OFFSET", to_string(0));
				kernel->AddDefineSubsitute(gradientPath, "WEIGHT_COLUMN_COUNT", to_string(inputDescription.TotalUnits()));

				deviceAndImageGradientKernels.insert(make_pair(device, kernel.get()));
				programs[device]->AttachKernel(move(kernel));
			}

			//Continuing with the back prop
			for (auto device : devices)
			{
				auto deviceInfo = device->DeviceInfo();

				unique_ptr<LayerKernel<T>> kernel(new LayerKernel<T>());

				kernel->AddIncludePath(OCLProgram::DefaultSourceLocation);
				kernel->AddSourcePath(backPropPath);
				kernel->SetKernelName("BackPerceptronKernel");

				//Now, let us query the device if we have enough memory to use constant weights / inputs / biases etc...
				auto maximumConstantBufferSize = deviceInfo.MaxConstantBufferSize();
				if (maximumConstantBufferSize > weights->ByteSize())
				{
					kernel->AddDefine(backPropPath, "CONSTANT_WEIGHTS");
					maximumConstantBufferSize -= weights->ByteSize();
				}

				auto byteSize = inBackMemDesc.TotalMemory() * sizeof(T);
				if (maximumConstantBufferSize > byteSize)
				{
					kernel->AddDefine(backPropPath, "CONSTANT_INPUT_DELTA");
					maximumConstantBufferSize -= byteSize;
				}

				byteSize = inForwardMemDesc.TotalMemory() * sizeof(T);
				if (maximumConstantBufferSize > byteSize)
				{
					kernel->AddDefine(backPropPath, "CONSTANT_INPUT");
					maximumConstantBufferSize -= byteSize;
				}

				kernel->AddGlobalSize(inForwardDataDesc.Width);
				kernel->AddGlobalSize(inForwardDataDesc.Height);
				kernel->AddGlobalSize(inForwardDataDesc.Units);

				kernel->AddDefineSubsitute(backPropPath, "OUTPUT_WIDTH_OFFSET", to_string(outBackMemDesc.WidthOffset));
				kernel->AddDefineSubsitute(backPropPath, "OUTPUT_HEIGHT_OFFSET", to_string(outBackMemDesc.HeightOffset));
				kernel->AddDefineSubsitute(backPropPath, "OUTPUT_UNIT_OFFSET", to_string(outBackMemDesc.UnitOffset));
				kernel->AddDefineSubsitute(backPropPath, "OUTPUT_STRIDE", to_string(outBackMemDesc.Width));
				kernel->AddDefineSubsitute(backPropPath, "OUTPUT_UNIT_ELEMENT_COUNT_INC_PADDING", to_string(outBackMemDesc.Width * outBackMemDesc.Height));
				kernel->AddDefineSubsitute(backPropPath, "INPUT_WIDTH_OFFSET", to_string(inForwardMemDesc.WidthOffset));
				kernel->AddDefineSubsitute(backPropPath, "INPUT_HEIGHT_OFFSET", to_string(inForwardMemDesc.HeightOffset));
				kernel->AddDefineSubsitute(backPropPath, "INPUT_UNIT_OFFSET", to_string(inForwardMemDesc.UnitOffset));
				kernel->AddDefineSubsitute(backPropPath, "INPUT_STRIDE", to_string(inForwardMemDesc.Width));
				kernel->AddDefineSubsitute(backPropPath, "INPUT_UNIT_ELEMENT_COUNT_INC_PADDING", to_string(inForwardMemDesc.Width * inForwardMemDesc.Height));
				kernel->AddDefineSubsitute(backPropPath, "INPUT_DELTA_OFFSET", to_string(inBackMemDesc.UnitOffset));
				kernel->AddDefineSubsitute(backPropPath, "INPUT_DELTA_LIMIT", to_string(inBackMemDesc.UnitOffset + outForwardDataDesc.Units));
				kernel->AddDefineSubsitute(backPropPath, "WEIGHT_COLUMN_COUNT", to_string(inForwardDataDesc.TotalUnits()));

				deviceAndImageBackKernels.insert(make_pair(device, kernel.get()));
				programs[device]->AttachKernel(move(kernel));
			}

			//Finally the forward prop
			for (auto device : devices)
			{
				auto deviceInfo = device->DeviceInfo();

				unique_ptr<LayerKernel<T>> kernel(new LayerKernel<T>());

				kernel->AddIncludePath(OCLProgram::DefaultSourceLocation);
				kernel->AddSourcePath(forwardPropPath);
				kernel->SetKernelName("ForwardPerceptronKernel");

				//In this case, we need to two additional kernels
				if (config.ActivationFunction() == MatunaSoftMaxActivation)
				{

					scalarCache = move(
						this->context->CreateMemory(CL_MEM_READ_WRITE, sizeof(T)));

					unique_ptr<LayerKernel<T>> simpleSumKernel(new LayerKernel<T>());
					simpleSumKernel->AddSourcePath(simpleSumPath);
					simpleSumKernel->AddIncludePath(OCLProgram::DefaultSourceLocation);
					simpleSumKernel->SetKernelName("SimpleSumKernel");
					simpleSumKernel->AddDefineSubsitute(simpleSumPath, "INPUT_COUNT", to_string(firstOutputData.TotalUnits()));

					deviceAndSimpleSumKernels.insert(make_pair(device, simpleSumKernel.get()));
					programs[device]->AttachKernel(move(simpleSumKernel));

					unique_ptr<LayerKernel<T>> divideByScalarKernel(new LayerKernel<T>());
					divideByScalarKernel->AddSourcePath(divideByScalarPath);
					divideByScalarKernel->AddIncludePath(OCLProgram::DefaultSourceLocation);
					divideByScalarKernel->SetKernelName("DivideByScalarKernel");
					divideByScalarKernel->AddGlobalSize(firstOutputData.TotalUnits());

					deviceAndDivideByScalarKernels.insert(make_pair(device, divideByScalarKernel.get()));
					programs[device]->AttachKernel(move(divideByScalarKernel));
				}

				//Now, let us query the device if we have enough memory to use constant weights / inputs / biases etc...
				auto maximumConstantBufferSize = deviceInfo.MaxConstantBufferSize();
				auto inputBytes = sizeof(T) * inForwardMemDesc.TotalMemory();
				if (maximumConstantBufferSize > inputBytes)
				{
					kernel->AddDefine(forwardPropPath, "CONSTANT_INPUT");
					maximumConstantBufferSize -= inputBytes;
				}
				if (maximumConstantBufferSize > weights->ByteSize())
				{
					kernel->AddDefine(forwardPropPath, "CONSTANT_WEIGHTS");
					maximumConstantBufferSize -= weights->ByteSize();
				}
				if (maximumConstantBufferSize > biases->ByteSize())
				{
					kernel->AddDefine(forwardPropPath, "CONSTANT_BIASES");
					maximumConstantBufferSize -= biases->ByteSize();
				}


				kernel->AddGlobalSize(outForwardDataDesc.TotalUnits());

				kernel->AddDefineSubsitute(forwardPropPath, "INPUT_UNITS_LIMIT", to_string(inForwardDataDesc.Units + inForwardMemDesc.UnitOffset));
				kernel->AddDefineSubsitute(forwardPropPath, "INPUT_WIDTH_LIMIT", to_string(inForwardDataDesc.Width + inForwardMemDesc.WidthOffset));
				kernel->AddDefineSubsitute(forwardPropPath, "INPUT_HEIGHT_LIMIT", to_string(inForwardDataDesc.Height +inForwardMemDesc.HeightOffset));
				kernel->AddDefineSubsitute(forwardPropPath, "INPUT_UNITS_OFFSET", to_string(inForwardMemDesc.UnitOffset));
				kernel->AddDefineSubsitute(forwardPropPath, "INPUT_WIDTH_OFFSET", to_string(inForwardMemDesc.WidthOffset));
				kernel->AddDefineSubsitute(forwardPropPath, "INPUT_HEIGHT_OFFSET", to_string(inForwardMemDesc.HeightOffset));
				kernel->AddDefineSubsitute(forwardPropPath, "COLUMN_COUNT", to_string(inForwardDataDesc.TotalUnits()));
				kernel->AddDefineSubsitute(forwardPropPath, "OUTPUT_UNIT_OFFSET", to_string(0));
				kernel->AddDefineSubsitute(forwardPropPath, "INPUT_UNIT_ELEMENT_COUNT_INC_PADDING", to_string(inForwardMemDesc.Width * inForwardMemDesc.Height));
				kernel->AddDefineSubsitute(forwardPropPath, "INPUT_MEMORY_WIDTH", to_string(inForwardMemDesc.Width));


				deviceAndImageForwardKernels.insert(make_pair(device, kernel.get()));
				programs[device]->AttachKernel(move(kernel));
			}
		}

		template<class T>
		void PerceptronLayer<T>::InitializeProgram(unordered_map<OCLDevice*, unique_ptr<OCLProgram>>& programs)
		{
			auto& firstOutputData = this->outForwardPropDataDescriptions[0];
			vector<OCLDevice*> devices = this->context->GetDevices();

			string gradientPath = Path::Combine(OCLProgram::DefaultSourceLocation, "GradientPerceptronKernel.cl");
			string backPropPath = Path::Combine(OCLProgram::DefaultSourceLocation, "BackPropPerceptronKernel.cl");
			string forwardPropPath = Path::Combine(OCLProgram::DefaultSourceLocation, "ForwardPerceptronKernel.cl");
			string simpleSumPath = Path::Combine(OCLProgram::DefaultSourceLocation, "SimpleSumKernel.cl");
			string divideByScalarPath = Path::Combine(OCLProgram::DefaultSourceLocation, "DivideByScalarKernel.cl");

			//Starting with the gradient kernel
			for (auto device : devices)
			{
				auto deviceInfo = device->DeviceInfo();

				unique_ptr<LayerKernel<T>> kernel(new LayerKernel<T>());

				kernel->AddIncludePath(OCLProgram::DefaultSourceLocation);
				kernel->AddSourcePath(gradientPath);
				kernel->SetKernelName("GradientPerceptronKernel");

				//Now, let us query the device if we have enough memory to use constant weights / inputs / biases etc...
				auto maximumConstantBufferSize = deviceInfo.MaxConstantBufferSize();
				auto byteSize = firstOutputData.TotalUnits() * sizeof(T);
				if (maximumConstantBufferSize > byteSize)
				{
					kernel->AddDefine(gradientPath, "CONSTANT_INPUT_DELTA");
					maximumConstantBufferSize -= byteSize;
				}

				byteSize = inputDescription.TotalUnits() * sizeof(T);
				if (maximumConstantBufferSize > byteSize)
				{
					kernel->AddDefine(gradientPath, "CONSTANT_INPUT");
					maximumConstantBufferSize -= byteSize;
				}

				kernel->AddGlobalSize(inputDescription.TotalUnits());
				kernel->AddGlobalSize(firstOutputData.TotalUnits());
				kernel->AddDefineSubsitute(gradientPath, "WEIGHT_COLUMN_COUNT", to_string(inputDescription.TotalUnits()));
				kernel->AddDefineSubsitute(gradientPath, "INPUT_DELTA_OFFSET", to_string(0));
				kernel->AddDefineSubsitute(gradientPath, "INPUT_OFFSET", to_string(0)); //We are not using these at the moment!

				deviceAndGradientKernels.insert(make_pair(device, kernel.get()));
				programs[device]->AttachKernel(move(kernel));
			}

			//Next back prop kernel
			for (auto device : devices)
			{
				auto deviceInfo = device->DeviceInfo();

				unique_ptr<LayerKernel<T>> kernel(new LayerKernel<T>());

				kernel->AddIncludePath(OCLProgram::DefaultSourceLocation);
				kernel->AddSourcePath(backPropPath);
				kernel->SetKernelName("BackPerceptronKernel");

				auto maximumConstantBufferSize = deviceInfo.MaxConstantBufferSize();
				if (maximumConstantBufferSize > weights->ByteSize())
				{
					kernel->AddDefine(backPropPath, "CONSTANT_WEIGHTS");
					maximumConstantBufferSize -= weights->ByteSize();
				}

				auto byteSize = firstOutputData.TotalUnits() * sizeof(T);
				if (maximumConstantBufferSize > byteSize)
				{
					kernel->AddDefine(backPropPath, "CONSTANT_INPUT_DELTA");
					maximumConstantBufferSize -= byteSize;
				}

				byteSize = inputDescription.TotalUnits() * sizeof(T);
				if (maximumConstantBufferSize > byteSize)
				{
					kernel->AddDefine(backPropPath, "CONSTANT_INPUT");
					maximumConstantBufferSize -= byteSize;
				}

				kernel->AddGlobalSize(inputDescription.TotalUnits());

				kernel->AddDefineSubsitute(backPropPath, "INPUT_DELTA_COUNT", to_string(firstOutputData.TotalUnits()));
				kernel->AddDefineSubsitute(backPropPath, "WEIGHT_COLUMN_COUNT", to_string(inputDescription.TotalUnits()));
				kernel->AddDefineSubsitute(backPropPath, "INPUT_OFFSET", to_string(0));
				kernel->AddDefineSubsitute(backPropPath, "INPUT_DELTA_OFFSET", to_string(0));
				kernel->AddDefineSubsitute(backPropPath, "OUTPUT_DELTA_OFFSET", to_string(0));

				deviceAndBackKernels.insert(make_pair(device, kernel.get()));
				programs[device]->AttachKernel(move(kernel));
			}

			//Finally, forward prop
			for (auto device : devices)
			{
				auto deviceInfo = device->DeviceInfo();

				//In this case, we need to two additional kernels
				if (config.ActivationFunction() == MatunaSoftMaxActivation)
				{

					scalarCache = move(
						this->context->CreateMemory(CL_MEM_READ_WRITE, sizeof(T)));

					unique_ptr<LayerKernel<T>> simpleSumKernel(new LayerKernel<T>());
					simpleSumKernel->AddSourcePath(simpleSumPath);
					simpleSumKernel->AddIncludePath(OCLProgram::DefaultSourceLocation);
					simpleSumKernel->SetKernelName("SimpleSumKernel");
					simpleSumKernel->AddDefineSubsitute(simpleSumPath, "INPUT_COUNT", to_string(firstOutputData.TotalUnits()));

					deviceAndSimpleSumKernels.insert(make_pair(device, simpleSumKernel.get()));
					programs[device]->AttachKernel(move(simpleSumKernel));

					unique_ptr<LayerKernel<T>> divideByScalarKernel(new LayerKernel<T>());
					divideByScalarKernel->AddSourcePath(divideByScalarPath);
					divideByScalarKernel->AddIncludePath(OCLProgram::DefaultSourceLocation);
					divideByScalarKernel->SetKernelName("DivideByScalarKernel");
					divideByScalarKernel->AddGlobalSize(firstOutputData.TotalUnits());

					deviceAndDivideByScalarKernels.insert(make_pair(device, divideByScalarKernel.get()));
					programs[device]->AttachKernel(move(divideByScalarKernel));
				}

				unique_ptr<LayerKernel<T>> kernel(new LayerKernel<T>());

				kernel->AddSourcePath(forwardPropPath);
				kernel->AddIncludePath(OCLProgram::DefaultSourceLocation);
				kernel->SetKernelName("ForwardPerceptronKernel");

				//Now, let us query the device if we have enough memory to use constant weights / inputs / biases etc...
				auto maximumConstantBufferSize = deviceInfo.MaxConstantBufferSize();
				auto inputBytes = sizeof(T) * inputDescription.TotalUnits();
				if (maximumConstantBufferSize > weights->ByteSize())
				{
					kernel->AddDefine(forwardPropPath, "CONSTANT_WEIGHTS");
					maximumConstantBufferSize -= weights->ByteSize();
				}
				if (maximumConstantBufferSize > inputBytes)
				{
					kernel->AddDefine(forwardPropPath, "CONSTANT_INPUT");
					maximumConstantBufferSize -= inputBytes;
				}
				if (maximumConstantBufferSize > biases->ByteSize())
				{
					kernel->AddDefine(forwardPropPath, "CONSTANT_BIASES");
					maximumConstantBufferSize -= biases->ByteSize();
				}

				kernel->AddGlobalSize(firstOutputData.TotalUnits());
				kernel->AddDefineSubsitute(forwardPropPath, "INPUT_COUNT", to_string(inputDescription.TotalUnits()));

				deviceAndForwardKernels.insert(make_pair(device, kernel.get()));
				programs[device]->AttachKernel(move(kernel));
			}
		}

		/*
		template<class T>
		void PerceptronLayer<T>::InitializeNormalGradientKernel()
		{
		//The memory maps to the data description in this case
		auto outputDataDescriptions = this->outForwardPropDataDescriptions;
		auto& firstOutputData = outputDataDescriptions[0];

		vector<OCLDevice*> devices = this->context->GetDevices();

		for (auto device : devices)
		{
		auto deviceInfo = device->DeviceInfo();

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
		vector<OCLDevice*> oneDeviceVector;
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

		vector<OCLDevice*> devices = this->context->GetDevices();

		for (auto device : devices)
		{
		auto deviceInfo = device->DeviceInfo();

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
		vector<OCLDevice*> oneDeviceVector;
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

		vector<OCLDevice*> devices = this->context->GetDevices();

		for (auto device : devices)
		{
		auto deviceInfo = device->DeviceInfo();

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
		vector<OCLDevice*> oneDeviceVector;
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

		vector<OCLDevice*> devices = this->context->GetDevices();

		for (auto device : devices)
		{
		auto deviceInfo = device->DeviceInfo();

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

		vector<OCLDevice*> oneDeviceVector;
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

		vector<OCLDevice*> devices = this->context->GetDevices();

		for (auto device : devices)
		{

		vector<OCLDevice*> oneDeviceVector;
		oneDeviceVector.push_back(device);
		auto deviceInfo = device->DeviceInfo();

		if (config.ActivationFunction() == MatunaSoftMaxActivation)
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
		void PerceptronLayer<T>::InitializeImageForwardPerceptron()
		{
		LayerDataDescription firstOutputData =
		this->outForwardPropDataDescriptions[0];
		LayerDataDescription firstInputData =
		this->InForwardPropDataDescriptions()[0];
		LayerMemoryDescription firstInputMemory =
		this->InForwardPropMemoryDescriptions()[0];

		int biasCount = firstOutputData.TotalUnits();

		vector<OCLDevice*> devices = this->context->GetDevices();

		for (auto device : devices)
		{

		vector<OCLDevice*> oneDeviceVector;
		oneDeviceVector.push_back(device);
		auto deviceInfo = device->DeviceInfo();

		if (config.ActivationFunction() == MatunaSoftMaxActivation)
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

		*/
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
			OCLDevice* device = this->context->GetDevices()[0];
			device->WriteMemory(weights.get(), sizeof(T) * initialWeightValues.size(),
				initialWeightValues.data(), 0, false);
			device->WriteMemory(biases.get(), sizeof(T) * initialBiasValues.size(),
				initialBiasValues.data(), 0, false);
			device->WaitForDeviceQueue(0);
		}

		template<class T>
		void PerceptronLayer<T>::EnqueueForwardPropagation(OCLDevice* device,
			int queueIndex, OCLMemory* previousInput, OCLMemory* output,
			bool blocking)
		{
			if (useImage)
			{
				auto& kernel = deviceAndImageForwardKernels[device];
				kernel->SetMemoryArg(previousInput, 0);
				kernel->SetMemoryArg(output, 1);
				if (config.ActivationFunction() == MatunaSoftMaxActivation)
				{
					device->ExecuteKernel(kernel, queueIndex, false);
					auto& sumKernel = deviceAndSimpleSumKernels[device];
					auto& scalarKernel = deviceAndDivideByScalarKernels[device];
					sumKernel->SetMemoryArg(output, 0);
					sumKernel->SetMemoryArg(scalarCache.get(), 1);
					device->ExecuteTask(sumKernel, queueIndex, false);
					scalarKernel->SetMemoryArg(output, 0);
					scalarKernel->SetMemoryArg(scalarCache.get(), 1);
					device->ExecuteKernel(scalarKernel, queueIndex, blocking);
				}
				else
					device->ExecuteKernel(kernel, queueIndex, blocking);
			}
			else
			{
				auto& kernel = deviceAndForwardKernels[device];
				kernel->SetMemoryArg(previousInput, 0);
				kernel->SetMemoryArg(output, 1);
				if (config.ActivationFunction() == MatunaSoftMaxActivation)
				{
					device->ExecuteKernel(kernel, queueIndex, false);
					auto& sumKernel = deviceAndSimpleSumKernels[device];
					auto& scalarKernel = deviceAndDivideByScalarKernels[device];
					sumKernel->SetMemoryArg(output, 0);
					sumKernel->SetMemoryArg(scalarCache.get(), 1);
					device->ExecuteTask(sumKernel, queueIndex, false);
					scalarKernel->SetMemoryArg(output, 0);
					scalarKernel->SetMemoryArg(scalarCache.get(), 1);
					device->ExecuteKernel(scalarKernel, queueIndex, blocking);
				}
				else
					device->ExecuteKernel(kernel, queueIndex, blocking);
			}
		}

		template<class T>
		void PerceptronLayer<T>::EnqueueBackPropagation(OCLDevice* device,
			int queueIndex, OCLMemory* previousInput, OCLMemory* delta,
			OCLMemory* deltaOutput, bool blocking)
		{
			if (useImage)
			{
				auto& kernel = deviceAndImageBackKernels[device];
				kernel->SetMemoryArg(previousInput, 0);
				kernel->SetMemoryArg(delta, 1);
				kernel->SetMemoryArg(deltaOutput, 2);
				device->ExecuteKernel(kernel, queueIndex, blocking);
			}
			else
			{
				auto& kernel = deviceAndBackKernels[device];
				kernel->SetMemoryArg(previousInput, 0);
				kernel->SetMemoryArg(delta, 1);
				kernel->SetMemoryArg(deltaOutput, 2);
				device->ExecuteKernel(kernel, queueIndex, blocking);
			}
		}

		template<class T>
		void PerceptronLayer<T>::EnqueueCalculateGradient(OCLDevice* device,
			int queueIndex, OCLMemory* previousInput, OCLMemory* delta,
			OCLMemory* gradient, bool blocking)
		{
			if (useImage)
			{
				auto& kernel = deviceAndImageGradientKernels[device];
				kernel->SetMemoryArg(previousInput, 0);
				kernel->SetMemoryArg(delta, 1);
				kernel->SetMemoryArg(gradient, 2);
				device->ExecuteKernel(kernel, queueIndex, blocking);
			}
			else
			{
				auto& kernel = deviceAndGradientKernels[device];
				kernel->SetMemoryArg(previousInput, 0);
				kernel->SetMemoryArg(delta, 1);
				kernel->SetMemoryArg(gradient, 2);
				device->ExecuteKernel(kernel, queueIndex, blocking);
			}

			//Since we don't need to calculate anything for the bias gradient, we simply use copy buffer.
			device->CopyCLMemory(delta, gradient, 0,
				config.Units() * inputDescription.TotalUnits() * sizeof(T),
				config.Units() * sizeof(T), queueIndex, blocking);
		}

		template<class T>
		void PerceptronLayer<T>::EnqueueCalculateGradient(OCLDevice* device, int queueIndex,
			OCLMemory* previousInput, OCLMemory* delta, vector<OCLMemory*> gradient, bool blocking)
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
				kernel->SetMemoryArg(previousInput, 0);
				kernel->SetMemoryArg(delta, 1);
				kernel->SetMemoryArg(gradient[0], 2);
				device->ExecuteKernel(kernel, queueIndex, blocking);
			}
			else
			{
				auto& kernel = deviceAndGradientKernels[device];
				kernel->SetMemoryArg(previousInput, 0);
				kernel->SetMemoryArg(delta, 1);
				kernel->SetMemoryArg(gradient[0], 2);
				device->ExecuteKernel(kernel, queueIndex, blocking);
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
		vector<OCLMemory*> PerceptronLayer<T>::GetParameters()
		{
			vector<OCLMemory*> result;
			result.push_back(weights.get());
			result.push_back(biases.get());
			return result;
		}

		template<class T>
		void PerceptronLayer<T>::GetParameters(T* parameters, OCLDevice* device,
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
		void PerceptronLayer<T>::SetParameters(T* parameters, OCLDevice* device,
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
} /* namespace Matuna */
