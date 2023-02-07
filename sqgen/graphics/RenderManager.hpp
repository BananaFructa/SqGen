#pragma once

/*
* modified code from the first SqGen simulator
*/

#include <SFML/Graphics.hpp>
#include <mutex>
#include <map>
#include "../Simulation.hpp"
#include "RenderUtils.hpp"

struct VertexBuffer {
private:
	sf::VertexArray FoodMapArray1 = sf::VertexArray(sf::Quads, 4 * Constants::totalMapSize);
	sf::VertexArray AgentMapArray1 = sf::VertexArray(sf::Quads, 4 * Constants::totalMapSize);
	sf::VertexArray FoodMapArray2 = sf::VertexArray(sf::Quads, 4 * Constants::totalMapSize);
	sf::VertexArray AgentMapArray2 = sf::VertexArray(sf::Quads, 4 * Constants::totalMapSize);

	bool use = false;

public:

	VertexBuffer() {
		genVertexArray(FoodMapArray1);
		genVertexArray(FoodMapArray2);
		genVertexArray(AgentMapArray1);
		genVertexArray(AgentMapArray2);
	}

	sf::VertexArray& getFrontFoodMap() {
		return (use ? FoodMapArray1 : FoodMapArray2);
	}

	sf::VertexArray& getBackFoodMap() {
		return (use ? FoodMapArray2 : FoodMapArray1);
	}

	sf::VertexArray& getFrontAgentMap() {
		return (use ? AgentMapArray1 : AgentMapArray2);
	}

	sf::VertexArray& getBackAgentMap() {
		return (use ? AgentMapArray2 : AgentMapArray1);
	}

	void swap() {
		use = !use;
	}
};

class RenderManager {

private:
	sf::Texture cursor;
	sf::Sprite cursorSprite;

	sf::Vector2i cursorPos = sf::Vector2i(0,0);

	std::map<SpecieID, sf::Color> colorPalette;

	sf::Vector2f LastCameraPosition = sf::Vector2f(0, 0);
	int XUnitsInFrame;
	int YUnitsInFrame;
	void FOVChanged();
	int AreaInFrame;
	bool SignalMapMode;

	bool paused = false;
	bool realFood = false;

public:

	bool shouldSave = false;

	RenderManager(Simulation& sim);

	sf::RenderWindow Window;
	sf::Vector2f CameraPosition = sf::Vector2f(0, 0);;
	float FieldOfView = 10;
	Simulation& SimulationToRender;

	VertexBuffer buffer;

	bool IsGridDisplayed = true;

	void updateRenderData();
	void RenderLoop();
	void RunEvent(sf::Event Event);

};