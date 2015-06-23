/*
* ConvNetFactoryVisitor.cpp
*
*  Created on: May 5, 2015
*      Author: Mikael
*/

#include "ConvNetFactoryVisitor.h"
#include "InterlockHelper.h"
#include "PerceptronLayerConfig.h"
#include "StandardOutputLayerConfig.h"
#include "ConvolutionLayerConfig.h"
#include "ConvNetConfig.h"
#include "ConvNet.h"
#include <stdexcept>
#include <type_traits>

namespace Matuna
{
	namespace MachineLearning
	{

		ConvNetFactoryVisitor::ConvNetFactoryVisitor(ConvNet* network) :
			network(network)
		{
			backPropActivation = MatunaLinearActivation;
			outputIsCalled = false;
		}

		ConvNetFactoryVisitor::~ConvNetFactoryVisitor()
		{

		}

		void ConvNetFactoryVisitor::InterlockLayer(BackPropLayer* layer)
		{
			if (!InterlockHelper::IsCompatible(layer->InForwardPropDataDescriptions(),
				layer->InForwardPropMemoryProposals()))
				throw runtime_error(
				"We have incompatible data with the memory proposal");

			if (!InterlockHelper::IsCompatible(layer->InBackPropDataDescriptions(),
				layer->InBackPropMemoryProposals()))
				throw runtime_error(
				"We have incompatible data with the memory proposal");

			if (!InterlockHelper::IsCompatible(layer->OutBackPropDataDescriptions(),
				layer->OutBackPropMemoryProposals()))
				throw runtime_error(
				"We have incompatible data with the memory proposal");

			if (!InterlockHelper::DataEquals(inputDataDescriptions,
				layer->OutBackPropDataDescriptions()))
				throw runtime_error("Invalid data description");
			if (!InterlockHelper::DataEquals(inputDataDescriptions,
				layer->InForwardPropDataDescriptions()))
				throw runtime_error("Invalid data description");

			if (!InterlockHelper::IsCompatible(layer->OutBackPropMemoryProposals(),
				backOutputProposals))
				throw runtime_error("Invalid memory description");

			if (!InterlockHelper::IsCompatible(layer->InForwardPropMemoryProposals(),
				forwardInputProposals))
				throw runtime_error("Invalid memory description");

			auto backPropOutputMemory = InterlockHelper::CalculateCompatibleMemory(
				layer->OutBackPropMemoryProposals(), backOutputProposals);

			auto forwardPropInput = InterlockHelper::CalculateCompatibleMemory(
				layer->InForwardPropMemoryProposals(), forwardInputProposals);

			layer->InterlockBackPropOutput(backPropOutputMemory);
			layer->InterlockForwardPropInput(forwardPropInput);

			//Finally we need to perform interlocking on the previous layer's output units
			if (layers.size() != 0)
			{
				auto& forwardPropLayer = layers[layers.size() - 1];

				if (!InterlockHelper::IsCompatible(
					forwardPropLayer->OutForwardPropDataDescriptions(),
					forwardPropLayer->OutForwardPropMemoryProposals()))
					throw runtime_error(
					"We have incompatible data with the memory proposal");

				if (!InterlockHelper::DataEquals(
					forwardPropLayer->InBackPropDataDescriptions(),
					forwardPropLayer->OutForwardPropDataDescriptions()))
					throw runtime_error(
					"The backprop input and forward output units are different");

				if (!InterlockHelper::IsCompatible(
					forwardPropLayer->InBackPropDataDescriptions(),
					forwardPropLayer->InBackPropMemoryProposals()))
					throw runtime_error(
					"We have incompatible data with the memory proposal");

				if (!InterlockHelper::DataEquals(
					forwardPropLayer->OutForwardPropDataDescriptions(),
					layer->InForwardPropDataDescriptions()))
					throw runtime_error(
					"The previous left output description doesn't match the right input description");

				if (!InterlockHelper::DataEquals(
					forwardPropLayer->InBackPropDataDescriptions(),
					layer->OutBackPropDataDescriptions()))
					throw runtime_error(
					"The previous left output description doesn't match the right input description");

				auto forwardPropOutputMemory =
					InterlockHelper::CalculateCompatibleMemory(
					forwardPropLayer->OutForwardPropMemoryProposals(),
					layer->InForwardPropMemoryProposals());

				auto backPropInputMemory = InterlockHelper::CalculateCompatibleMemory(
					forwardPropLayer->InBackPropMemoryProposals(),
					layer->OutBackPropMemoryProposals());

				forwardPropLayer->InterlockForwardPropOutput(forwardPropOutputMemory);
				forwardPropLayer->InterlockBackPropInput(backPropInputMemory);

				if (!forwardPropLayer->Interlocked())
					throw runtime_error(
					"The back-forward prop layer is not interlocked");

				//Need to call this to indicate to the layer that the interlock has been finalized
				forwardPropLayer->InterlockFinalized();
			}
			else //Interlock the ConvNet here
			{
				if (!InterlockHelper::DataEquals(network->InputForwardDataDescriptions(),
					layer->OutBackPropDataDescriptions()))
					throw runtime_error(
					"The previous left output description doesn't match the right input description");

				if (!InterlockHelper::DataEquals(network->InputForwardDataDescriptions(),
					layer->InForwardPropDataDescriptions()))
					throw runtime_error(
					"The previous left output description doesn't match the right input description");

				network->InterlockForwardPropInput(layer->InForwardPropMemoryDescriptions());
			}
		}

		void ConvNetFactoryVisitor::InterlockLayer(ForwardBackPropLayer* layer)
		{
			static_assert(is_base_of<BackPropLayer, ForwardBackPropLayer>::value,
				"ForwardBackPropLayer is not a sub class of BackPropLayer");

			InterlockLayer(dynamic_cast<BackPropLayer*>(layer));

			forwardInputProposals = layer->OutForwardPropMemoryProposals();
			backOutputProposals = layer->InBackPropMemoryProposals();
			inputDataDescriptions = layer->OutForwardPropDataDescriptions();
		}

		void ConvNetFactoryVisitor::InterlockAndAddLayer(const PerceptronLayerConfig* const config, unique_ptr<ForwardBackPropLayer> layer)
		{
			if (outputIsCalled)
				throw invalid_argument("The output layer has already been created. We cannot proceed");

			backPropActivation = config->ActivationFunction();
			InterlockLayer(layer.get());
			layers.push_back(move(layer));
		}

		void ConvNetFactoryVisitor::InterlockAndAddLayer(const VanillaSamplingLayerConfig* const, unique_ptr<ForwardBackPropLayer> layer)
		{
			if (outputIsCalled)
				throw invalid_argument("The output layer has already been created. We cannot proceed");

			backPropActivation = MatunaLinearActivation;
			InterlockLayer(layer.get());
			layers.push_back(move(layer));
		}

		void ConvNetFactoryVisitor::InterlockAndAddLayer(const MaxPoolingLayerConfig* const, unique_ptr<ForwardBackPropLayer> layer)
		{
			if (outputIsCalled)
				throw invalid_argument("The output layer has already been created. We cannot proceed");

			backPropActivation = MatunaLinearActivation;
			InterlockLayer(layer.get());
			layers.push_back(move(layer));
		}

		void ConvNetFactoryVisitor::InterlockAndAddLayer(const ConvolutionLayerConfig* const config, unique_ptr<ForwardBackPropLayer> layer)
		{
			if (outputIsCalled)
				throw invalid_argument("The output layer has already been created. We cannot proceed");

			backPropActivation = config->ActivationFunction();
			InterlockLayer(layer.get());
			layers.push_back(move(layer));
		}

		void ConvNetFactoryVisitor::InterlockAndAddLayer(const StandardOutputLayerConfig* const, unique_ptr<OutputLayer> layer)
		{
			if (outputIsCalled)
				throw invalid_argument("The output layer has already been created. We cannot proceed");

			InterlockLayer(layer.get());

			//Since this is an output layer, this will define the targets.
			//We could potentially have some value in the config if we want to do something about this.
			layer->InterlockBackPropInput(layer->InBackPropMemoryProposals());

			if (!layer->Interlocked())
				throw runtime_error("The output layer is not interlocked");

			layer->InterlockFinalized();

			network->InterlockForwardPropDataOutput(layer->InBackPropDataDescriptions());
			network->InterlockForwardPropOutput(layer->InBackPropMemoryDescriptions());

			if (layers.size() != 0)
			{
				auto& firstLayer = layers[0];
				network->InterlockBackPropOutput(firstLayer->InBackPropMemoryDescriptions());
				network->InterlockBackPropDataOutput(firstLayer->InBackPropDataDescriptions());
			}
			else
			{
				network->InterlockBackPropOutput(layer->InBackPropMemoryDescriptions());
				network->InterlockBackPropDataOutput(layer->InBackPropDataDescriptions());
			}

			if (!network->Interlocked())
				throw runtime_error("The network is not interlocked");

			outputLayer = move(layer);

			outputIsCalled = true;
		}

		void ConvNetFactoryVisitor::InitializeInterlock(const ConvNetConfig* const config)
		{
			if (outputIsCalled)
				throw invalid_argument("The output layer has already been created. We cannot proceed");

			auto inputData = config->InputDataDescription();

			vector<LayerMemoryDescription> inputMemory;
			for (auto& data : inputData)
			{
				LayerMemoryDescription memory;
				memory.Width = data.Width;
				memory.Height = data.Height;
				memory.Units = data.Units;
				memory.HeightOffset = 0;
				memory.WidthOffset = 0;
				memory.UnitOffset = 0;
				inputMemory.push_back(memory);
			}

			forwardInputProposals = inputMemory;
			backOutputProposals = inputMemory;
			inputDataDescriptions = inputData;
		}

		vector<unique_ptr<ForwardBackPropLayer>> ConvNetFactoryVisitor::GetLayers()
		{
			vector<unique_ptr<ForwardBackPropLayer>> result;

			for (auto& layer : layers)
				result.push_back(move(layer));

			layers.clear();

			return result;
		}

		unique_ptr<OutputLayer> ConvNetFactoryVisitor::GetOutputLayer()
		{
			return move(outputLayer);
		}

	} /* namespace MachineLearning */
} /* namespace Matuna */
