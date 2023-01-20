#pragma once

#include <map>
#include <vector>
#include "Agent.hpp"
#include "../ha_models/NNModel.hpp"
#include "../ha_models/ReferenceMappedTensor.hpp"
#include "../ha_models/TensorMemAllocator.hpp"

struct MutationCfg {
	float nonZeroProb;
	float nonZeroLow;
	float nonZeroHigh;
	float zeroProb;
	float zeroHigh;
	float zeroLow;
};

struct NNAgentModelManager {
private:
public:

	bool hasVariables = false; // It wouldn't make sense for this to be false wouldn't it ? But it will be here for consistency
	bool hasStates = false;

	size_t poolSize = 0;
	size_t paramCount = 0;
	size_t stateCount = 0;

	size_t outputParamCount = 0;

	// Would have used the tensor allocator must is more efficient this way

	std::vector<Size> variableSizes;
	std::vector<Tensor*> preAllocatedVariables;
	std::map<SpecieID, Tensor*> agentModelVariables;
	
	std::vector<Size> stateSizes;
	std::vector<Tensor*> preAllocatedStates;
	std::map<AgentResourceID, Tensor*> agentModelState;

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
	void compile(Agent agents[], size_t agentCount, size_t specieRepeat = 1);
	Tensor predict(Tensor& input);

	void eraseSpecie(SpecieID id);
	void registerSpecie(SpecieID parent, SpecieID id, float prob, float low, float high, float zprob);
	void registerNewSpiece(SpecieID id, float low, float high);
	void registerNewSpiece(SpecieID id, size_t inputUse, size_t hiddenUse, float low, float high);

	void eraseAgent(AgentResourceID id);
	void registerAgent(AgentResourceID id);

	void loadSpiecie(const char* path, SpecieID id);
	void loadState(const char* path, AgentResourceID id);

	void cleanup();
	void free();

};