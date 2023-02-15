#include <thread>
#include <future>
#include <ppl.h>
#include "RenderManager.hpp"
#include "RenderUtils.hpp"

void RenderManager::updateRenderData() {

    float max = 0;

    float* foodMap = SimulationToRender.getFoodMap();

    SpecieID* specieMap = SimulationToRender.getSpecieMap();
    for (size_t l = 0; l < Constants::totalMapSize; l++) {
        size_t i = l / Constants::mapSize;
        size_t j = l % Constants::mapSize;
        max = std::max(foodMap[j + i * Constants::mapSize], max);
    }

    if (paused) return;

    float* signalMap = SimulationToRender.getSignalMap();
    float* attackMap = SimulationToRender.getAttackMap();

    sf::VertexArray& backFood = buffer.getBackFoodMap();
    sf::VertexArray& backAgent = buffer.getBackAgentMap();

    //concurrency::parallel_for((size_t)0, Constants::totalMapSize, [&](size_t l) {
    for (size_t l = 0;l < Constants::totalMapSize;l++) {
        size_t i = l / Constants::mapSize;
        size_t j = l % Constants::mapSize;

        float factor;
        if (realFood) {
            factor = SimulationToRender.getMediumAt(Position2i(i,j)).toFloat() / (2 * Constants::mediumInitial.toFloat());
        }
        else {
            factor = foodMap[j + i * Constants::mapSize];// / Constants::FinitialMapFood;
        }
        factor = std::max(0.0f,std::min(1.0f, factor));
        sf::Color Color = ( !realFood ? sf::Color::Color(13 * factor, 140 * factor, 5 * factor) : sf::Color::Color(0, 102 * factor, 255 * factor));
        backFood[(j + i * Constants::mapSize) * 4].color = Color;
        backFood[(j + i * Constants::mapSize) * 4 + 1].color = Color;
        backFood[(j + i * Constants::mapSize) * 4 + 2].color = Color;
        backFood[(j + i * Constants::mapSize) * 4 + 3].color = Color;
    }//);

    if (!SignalMapMode && !attackMapMode) {
        concurrency::parallel_for((size_t)0, Constants::totalMapSize, [&](size_t l) {
            int x = l / Constants::mapSize;
            int y = l % Constants::mapSize;
            SpecieID id = specieMap[y + x * Constants::mapSize];
            sf::Color color(0, 0, 0, 0);
            if (id != NULL_ID) {
                Color c = SimulationToRender.specieColorPallete[id];
                color = sf::Color(c.r, c.g, c.b);
            }
            backAgent[(y + x * Constants::mapSize) * 4].color = color;
            backAgent[(y + x * Constants::mapSize) * 4 + 1].color = color;
            backAgent[(y + x * Constants::mapSize) * 4 + 2].color = color;
            backAgent[(y + x * Constants::mapSize) * 4 + 3].color = color;
        });
    }
    else {
        concurrency::parallel_for((size_t)0, Constants::totalMapSize, [&](size_t l) {
            int x = l / Constants::mapSize;
            int y = l % Constants::mapSize;
            float s = (SignalMapMode ? signalMap[y + x * Constants::mapSize] : attackMap[y + x * Constants::mapSize]);
            float factor = (s + 1) / 2;
            sf::Color color(factor * 255, 0, (1 - factor) * 255);

            backAgent[(y + x * Constants::mapSize) * 4].color = color;
            backAgent[(y + x * Constants::mapSize) * 4 + 1].color = color;
            backAgent[(y + x * Constants::mapSize) * 4 + 2].color = color;
            backAgent[(y + x * Constants::mapSize) * 4 + 3].color = color;
        });
    }

    buffer.swap();

}

void RenderManager::FOVChanged() {
    XUnitsInFrame = XUnitsCount(Window,FieldOfView);
    YUnitsInFrame = YUnitsCount(Window, FieldOfView);
}

RenderManager::RenderManager(Simulation& sim) : SimulationToRender(sim), Window(sf::VideoMode(800, 800), "SqGen") {
    cursor.loadFromFile("tex/curs.png");
    cursorSprite.setTexture(cursor);
    cursorSprite.scale(sf::Vector2f(1.0f / 30, 1.0f / 30));
    updateRenderData();
    colorPalette[NULL_ID] = sf::Color::Transparent;
}

void RenderManager::RenderLoop() {

    FOVChanged();

    Window.setFramerateLimit(30);

    while (Window.isOpen()) {

        LastCameraPosition = CameraPosition;

        sf::Event event_;

        while (Window.pollEvent(event_)) {
            RunEvent(event_);
        }

        Window.clear();

        NormalizeViewport(Window, FieldOfView, CameraPosition);

        Window.draw(buffer.getFrontFoodMap());
        Window.draw(buffer.getFrontAgentMap());

        NormalizeViewport(Window, FieldOfView, sf::Vector2f(Fract(CameraPosition.x), Fract(CameraPosition.y)));

        IsGridDisplayed = FieldOfView > 7;

        if (IsGridDisplayed)
            DrawGrid(Window, FieldOfView);

        cursorSprite.setPosition((sf::Vector2f)cursorPos - sf::Vector2f(0.05 + Constants::mapSize / 2,0.05 + Constants::mapSize / 2) - CameraPosition);
        Window.draw(cursorSprite);

        Window.display();
    }
}

void RenderManager::RunEvent(sf::Event Event) {
    if (Event.type == sf::Event::Closed)
        Window.close();

    if (sf::Keyboard::isKeyPressed(sf::Keyboard::T)) {
        shouldSave = true;
    }

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

    if (Event.type == sf::Event::MouseWheelMoved) {
        if (Event.mouseWheel.delta > 0) FieldOfView += 0.2f;
        if (Event.mouseWheel.delta < 0) FieldOfView -= 0.2f;
        FOVChanged();
    }

    if (Event.type == sf::Event::MouseButtonPressed) {
        sf::Vector2i pixelPos = sf::Mouse::getPosition(Window);
        sf::Vector2f worldPos = Window.mapPixelToCoords(pixelPos) + CameraPosition;

        cursorPos = sf::Vector2i((int)(worldPos.x + Constants::mapSize/2), (int)(worldPos.y + Constants::mapSize / 2));
        
        if (cursorPos.x < 0 || cursorPos.y >= Constants::mapSize || cursorPos.y < 0 || cursorPos.y >= Constants::mapSize) return;

        if (SimulationToRender.getSpecieMap()[cursorPos.y + cursorPos.x * Constants::mapSize] != NULL_ID) {
            int a = 9;
            std::async([&]() {

                std::cout << "=========================\n";

                Agent agent = SimulationToRender.getAgents()[SimulationToRender.getAgentAt(Position2i(cursorPos.x, cursorPos.y))];

                float sig[Constants::spicieSignalCount];

                SimulationToRender.getSignalDict()[agent.specieId].getValue(sig);

                std::cout << "Specie signal:\n";
                for (size_t i = 0; i < Constants::spicieSignalCount; i++) {
                    std::cout << sig[i] << " ";
                }

                std::cout << '\n';

                std::cout << "Food level: " << agent.food.toFloat() << '\n';
                std::cout << "Current sigal: " << SimulationToRender.getSignalMap()[cursorPos.y + cursorPos.x * Constants::mapSize] << "\n";

                std::cout << "Generation: " << agent.generation << "\n";
                std::cout << "Specie ID: " << agent.specieId << "\n";
                std::cout << "Agent ID: " << agent.id << "\n";

                std::cout << "=========================\n";

            });

        }
    }

    if (Event.type == sf::Event::KeyPressed) {
        if (Event.key.code == sf::Keyboard::O) SignalMapMode = !SignalMapMode;
        if (Event.key.code == sf::Keyboard::P) SimulationToRender.togglePause();
        if (Event.key.code == sf::Keyboard::I) paused = !paused;
        if (Event.key.code == sf::Keyboard::U) realFood = !realFood;
        if (Event.key.code == sf::Keyboard::Y) SimulationToRender.step = true;
        if (Event.key.code == sf::Keyboard::R) attackMapMode = !attackMapMode;
    }
}