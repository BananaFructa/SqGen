#include "Simulation.hpp"

void Simulation::buildSIE(NNModel& model) {
	// TODO: build SIE model arhitecture
}

void Simulation::buildAP(NNModel& model) {
	// TODO: build AP model arhitecture
}

Simulation::Simulation() {

	buildSIE(SIE_Network);
	buildAP(AP_Netowrk);

	SIE_Manager = NNAgentModelManager(SIE_Network, curandManager);
	AP_Manager = NNAgentModelManager(AP_Netowrk, curandManager);

}
