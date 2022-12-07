#include "NNAgentModelManager.hpp"

Tensor* NNAgentModelManager::getVariableSet() {

	bool reuse = !preAllocatedVariables.empty();
	Tensor* set = reuse ? preAllocatedVariables[preAllocatedVariables.size() - 1] : new Tensor[variableSizes.size()];
	if (reuse) preAllocatedVariables.pop_back();
	else {
		for (size_t i = 0; i < variableSizes.size(); i++) {
			set[i].init(variableSizes[i]);
		}
	}
	return set;

}

Tensor* NNAgentModelManager::getStateSet() {

	bool reuse = !preAllocatedStates.empty();
	Tensor* set = reuse ? preAllocatedStates[preAllocatedStates.size() - 1] : new Tensor[stateSizes.size()];
	if (reuse) preAllocatedStates.pop_back();
	else {
		for (size_t i = 0; i < stateSizes.size(); i++) {
			set[i].init(stateSizes[i]);
		}
	}

	return set;

}


NNAgentModelManager::NNAgentModelManager() {
}

NNAgentModelManager::NNAgentModelManager(NNModel model, CurandManager manager) {

	// Copy the supermodel arhitecture
	this->curandManager = manager;
	this->supermodel = model;
	this->poolSize = model.poolSize;
	this->variableCount = model.variableCount;
	this->stateCount = model.stateCount;
	this->hasVariables = variableCount > 0;
	this->hasStates = stateCount > 0;

	// Reserve vecotr for model variable tensors
	variableSizes = std::vector<Size>(model.variableCount);
	stateSizes = std::vector<Size>(model.stateCount);

	compiledData = std::vector<ReferenceMappedTensor>(model.variableCount);
	compiledState = std::vector<ReferenceMappedTensor>(model.stateCount);

	// Get layer vector
	std::vector<Layer*>& layers = model.getLayers();
	// Variable to keep track of the current variable tensor
	size_t current = 0;

	// Iterate through layers and get the sizes of each variable tensor
	if (hasVariables) {
		for (size_t i = 0; i < layers.size(); i++) {
			if (layers[i]->getParamCount() == 0) continue;
			layers[i]->getParamsSizes(&variableSizes[current]);
			current += layers[i]->getParamCount();
		}
	}

	if (hasVariables) outputParamCount = layers[layers.size() - 1]->getParamCount();

	current = 0;

	// Iterate through layers and get the sizes of each state tensor
	if (hasStates) {
		for (size_t i = 0; i < layers.size(); i++) {
			if (layers[i]->getStateCount() == 0) continue;
			layers[i]->getStateSizes(&stateSizes[current]);
			current += layers[i]->getStateCount();
		}
	}

	// Get the input size end extend with the pool (batch) size
	inputSize = Size(layers[0]->getInputSize());
	inputSize.extend(poolSize);

	for (int i = 0; i < model.variableCount; i++) {
		Size s = Size(variableSizes[i]);
		s.extend(poolSize);
		compiledData[i] = ReferenceMappedTensor(s);
	}

	for (int i = 0; i < model.stateCount; i++) {
		Size s = Size(stateSizes[i]);
		s.extend(poolSize);
		compiledState[i] = ReferenceMappedTensor(s);
	}

}

void NNAgentModelManager::compile(Agent agents[], size_t agentCount) {

	for (int i = 0; i < agentCount; i++) {
		if (hasVariables) {
			for (int j = 0; j < supermodel.variableCount; j++) {
				compiledData[j].setRef(i, agentModelVariables[agents[i].specieId][j]);
			}
		}
		if (hasStates) {
			for (int j = 0; j < supermodel.stateCount; j++) {
				compiledState[j].setRef(i, agentModelState[agents[i].id][j]);
			}
		}
	}

	std::vector<Tensor> slicedData(supermodel.variableCount);
	std::vector<Tensor> slicedState(supermodel.stateCount);

	if (hasVariables) {
		for (int i = 0; i < supermodel.variableCount; i++) {
			compiledData[i].syncMap();
			slicedData[i] = compiledData[i].slice(0, agentCount);
		}
	}

	if (hasStates) {
		for (int i = 0; i < supermodel.stateCount; i++) {
			compiledState[i].syncMap();
			slicedState[i] = compiledState[i].slice(0, agentCount);
		}
	}

	if (hasVariables) supermodel.loadModel(&slicedData[0]);
	if (hasStates) supermodel.loadState(&slicedState[0]);

}

Tensor NNAgentModelManager::predict(Tensor& input) {
	supermodel.predict(input);
	return supermodel.getPrediction();
}

void NNAgentModelManager::eraseSpecie(SpecieID id) {

	if (!hasVariables) return;

	preAllocatedVariables.push_back(agentModelVariables[id]); // Register old data to be recycled
	agentModelVariables.erase(id);

}

void NNAgentModelManager::registerSpecie(SpecieID parent, SpecieID id, float prob, float low, float high, float zprob) {

	if (!hasVariables) return;

	Tensor* tensorsData = getVariableSet();

	Tensor* parentData = agentModelVariables[parent];

	for (size_t i = 0; i < variableSizes.size(); i++) {

		parentData[i].copyTo(tensorsData[i]);
		curandManager.rndOffsetTensorUniform(tensorsData[i], prob, low, high, zprob);

	}

	agentModelVariables[id] = tensorsData;
}

void NNAgentModelManager::registerNewSpiece(SpecieID id, float low, float high) {
	if (!hasVariables) return;

	Tensor* tensorData = getVariableSet();

	for (size_t i = 0; i < variableCount; i++) {

		curandManager.randomizeTensorUniform(tensorData[i], low, high);

	}

	agentModelVariables[id] = tensorData;
}

void NNAgentModelManager::registerNewSpiece(SpecieID id, size_t inputUse, size_t hiddenUse, float low, float high) {

	if (!hasVariables) return;

	Tensor* tensorData = getVariableSet();

	for (size_t i = 0; i < variableCount; i++) {

		Tensor sliced;

		if (i < variableCount - outputParamCount) {
			sliced = tensorData[i].slice(0, hiddenUse);
			tensorData[i].slice(hiddenUse, tensorData[i].size.last()).initZero();
		}
		else {
			sliced = tensorData[i];
		}

		if (i == 0) {
			Tensor subSlice;
			for (size_t j = 0; j < sliced.size.getDimSize(1); j++) {
				subSlice = sliced.slice(j, j + 1).squeeze().slice(0, inputUse);
				sliced.slice(inputUse, sliced.size.last()).initZero();
				curandManager.randomizeTensorUniform(subSlice, low, high);
			}
		}
		else {
			curandManager.randomizeTensorUniform(sliced, low, high);
		}

	}

	agentModelVariables[id] = tensorData;

}

void NNAgentModelManager::eraseAgent(AgentID id) {

	if (!hasStates) return;

	preAllocatedStates.push_back(agentModelState[id]);
	agentModelState.erase(id);

}

void NNAgentModelManager::registerAgent(AgentID id) {

	if (!hasStates) return;

	Tensor* tensorData = getStateSet();

	for (size_t i = 0; i < stateSizes.size(); i++) {

		tensorData[i].initZero();

	}

	agentModelState[id] = tensorData;

}

void NNAgentModelManager::loadSpiecie(SpecieID id, Tensor values[]) {
	// TODO: Implementation
}

void NNAgentModelManager::loadState(Agent id, Tensor states[]) {
	// TODO: Implementation
}

void NNAgentModelManager::cleanup() {
	for (size_t i = 0; i < preAllocatedVariables.size(); i++) {
		for (size_t j = 0; j < variableSizes.size(); i++) {
			preAllocatedVariables[i][j].free();
		}
		delete[] preAllocatedVariables[i];
	}
	preAllocatedVariables.clear();
	for (size_t i = 0; i < preAllocatedStates.size(); i++) {
		for (size_t j = 0; j < stateSizes.size(); j++) {
			preAllocatedStates[i][j].free();
		}
		delete[] preAllocatedStates[i];
	}
	preAllocatedStates.clear();
}

void NNAgentModelManager::free() {
	cleanup();
	// TODO: erase existing maps
}