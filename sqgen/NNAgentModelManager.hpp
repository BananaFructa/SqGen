#pragma once

#include <map>
#include <vector>
#include "Agent.hpp"
#include "../ha_models/NNModel.hpp"
#include "../ha_models/ReferenceMappedTensor.hpp"

struct NNAgentModelManager {
private:

	bool hasVariables = false; // It wouldn't make sense for this to be false wouldn't it ? But it will be here for consistency
	bool hasStates = false;

	size_t poolSize = 0;
	size_t variableCount = 0;
	size_t stateCount = 0;

	std::vector<Size> variableSizes;
	std::vector<Tensor*> preAllocatedVariables;
	std::map<SpecieID, Tensor*> agentModelVariables;
	
	std::vector<Size> stateSizes;
	std::vector<Tensor*> preAllocatedStates;
	std::map<AgentID, Tensor*> agentModelState;

	NNModel supermodel;

	CurandManager curandManager;

	std::vector<ReferenceMappedTensor> compiledData;
	std::vector<ReferenceMappedTensor> compiledState;

	Tensor* getVariableSet();
	Tensor* getStateSet();

public:

	Size inputSize;

	NNAgentModelManager();
	NNAgentModelManager(NNModel model,CurandManager curandmanager);
	void compile(Agent agents[], size_t agentCount);
	Tensor predict(Tensor& input);

	void eraseSpecie(SpecieID id);
	void registerSpecie(SpecieID parent, SpecieID id, float prob, float low, float high);
	void registerNewSpiece(SpecieID id, float low, float high);

	void eraseAgent(AgentID id);
	void registerAgent(AgentID id);

	void loadSpiecie(SpecieID id, Tensor values[]);
	void loadState(Agent id, Tensor states[]);

	void cleanup();
	void free();

};