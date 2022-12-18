#include <thread>
#include "RenderManager.hpp"
#include "RenderUtils.hpp"

void RenderManager::updateRenderData() {
    float* foodMap = SimulationToRender.getFoodMap();
    float* signalMap = SimulationToRender.getSignalMap();

    sf::VertexArray& backFood = buffer.getBackFoodMap();
    sf::VertexArray& backAgent = buffer.getBackAgentMap();

    for (int i = 0; i < Constants::mapSize; ++i) {
        for (int j = 0; j < Constants::mapSize; ++j) {
            float factor = foodMap[j + i * Constants::mapSize] / Constants::initialMapFood;
            sf::Color Color = sf::Color::Color(13 * factor, 140 * factor, 5 * factor);
            backFood[(j + i * Constants::mapSize) * 4].color = Color;
            backFood[(j + i * Constants::mapSize) * 4 + 1].color = Color;
            backFood[(j + i * Constants::mapSize) * 4 + 2].color = Color;
            backFood[(j + i * Constants::mapSize) * 4 + 3].color = Color;
        }
    }

    // Draw agents and create coloring system

    buffer.swap();

}

void RenderManager::FOVChanged() {
    XUnitsInFrame = XUnitsCount(Window,FieldOfView);
    YUnitsInFrame = YUnitsCount(Window, FieldOfView);
}

RenderManager::RenderManager(Simulation& sim) : SimulationToRender(sim), Window(sf::VideoMode(800, 800), "test") {
    updateRenderData();
}

void RenderManager::RenderLoop() {

    FOVChanged();
    UpdateFoodMapColors();

    Window.setFramerateLimit(30);

    while (Window.isOpen()) {

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::W)) {
            CameraPosition.y -= 2.0f;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::A)) {
            CameraPosition.x -= 2.0f;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::S)) {
            CameraPosition.y += 2.0f;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::D)) {
            CameraPosition.x += 2.0f;
        }

        LastCameraPosition = CameraPosition;

        sf::Event event_;

        while (Window.pollEvent(event_)) {
            RunEvent(event_);
        }

        Window.clear();

        NormalizeViewport(Window, FieldOfView, CameraPosition);

        Window.draw(buffer.getFrontFoodMap());
       // Window.draw(buffer.getFrontAgentMap());

        NormalizeViewport(Window, FieldOfView, sf::Vector2f(Fract(CameraPosition.x), Fract(CameraPosition.y)));

        IsGridDisplayed = FieldOfView > 7;

        if (IsGridDisplayed)
            DrawGrid(Window, FieldOfView);

        Window.display();
    }
}

void RenderManager::RunEvent(sf::Event Event) {
    if (Event.type == sf::Event::Closed)
        Window.close();

    if (Event.type == sf::Event::MouseWheelMoved) {
        if (Event.mouseWheel.delta > 0) FieldOfView++;
        if (Event.mouseWheel.delta < 0) FieldOfView--;
        FOVChanged();
    }

    if (Event.type == sf::Event::KeyPressed) {
        if (Event.key.code == sf::Keyboard::P) SignalMapMode = !SignalMapMode;
        if (Event.key.code == sf::Keyboard::O) AttackMapMode = !AttackMapMode;
    }
}

void RenderManager::UpdateFoodMapColors() {

    /*int index = 0;
    for (int i = 0; i < SimulationToRender->MapSize; ++i) {
        for (int j = 0; j < SimulationToRender->MapSize; ++j) {
            float factor = (float)((SimulationToRender->Map[i][j].Energy) / (float)Constants::MAX_ENERGY_IN_GENERATED_TILE);
            sf::Color Color = sf::Color::Color(13 * factor, 140 * factor, 5 * factor);
            float x = i - SimulationToRender->MapSize / 2;
            float y = j - SimulationToRender->MapSize / 2;
            FoodMapArray[index++].color = Color;
            FoodMapArray[index++].color = Color;
            FoodMapArray[index++].color = Color;
            FoodMapArray[index++].color = Color;
        }
    }*/
}
