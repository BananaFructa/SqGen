#include <thread>
#include "RenderManager.hpp"
#include "RenderUtils.hpp"

void RenderManager::updateRenderData() {

    float* foodMap = SimulationToRender.getFoodMap();
    float* signalMap = SimulationToRender.getSignalMap();
    SpecieID* specieMap = SimulationToRender.getSpecieMap();

    sf::VertexArray& backFood = buffer.getBackFoodMap();
    sf::VertexArray& backAgent = buffer.getBackAgentMap();

    for (int i = 0; i < Constants::mapSize; ++i) {
        for (int j = 0; j < Constants::mapSize; ++j) {

            {
                SpecieID id = specieMap[j + i * Constants::mapSize];
                sf::Color color;
                if (id != 0 && !colorPalette.count(id)) {
                    SpecieID parent = SimulationToRender.getParentSpecie(id);
                    if (parent == NULL_ID) color = randomAgentColor();
                    else color = mutateColor(colorPalette[parent]);
                    colorPalette[id] = color;
                }
            }

            float factor = foodMap[j + i * Constants::mapSize] / Constants::initialMapFood;
            sf::Color Color = sf::Color::Color(13 * factor, 140 * factor, 5 * factor);
            backFood[(j + i * Constants::mapSize) * 4].color = Color;
            backFood[(j + i * Constants::mapSize) * 4 + 1].color = Color;
            backFood[(j + i * Constants::mapSize) * 4 + 2].color = Color;
            backFood[(j + i * Constants::mapSize) * 4 + 3].color = Color;
        }
    }

    if (!SignalMapMode) {
        for (int x = 0; x < Constants::mapSize; x++) {
            for (int y = 0; y < Constants::mapSize; y++) {
                SpecieID id = specieMap[y + x * Constants::mapSize];
                sf::Color color = colorPalette[id];

                backAgent[(y + x * Constants::mapSize) * 4].color = color;
                backAgent[(y + x * Constants::mapSize) * 4 + 1].color = color;
                backAgent[(y + x * Constants::mapSize) * 4 + 2].color = color;
                backAgent[(y + x * Constants::mapSize) * 4 + 3].color = color;
            }
        }
    }
    else {
        for (int x = 0; x < Constants::mapSize; x++) {
            for (int y = 0; y < Constants::mapSize; y++) {
                float s = signalMap[y + x * Constants::mapSize];
                float factor = (s + 1) / 2;
                sf::Color color((1-factor)*255, 0, factor * 255);

                backAgent[(y + x * Constants::mapSize) * 4].color = color;
                backAgent[(y + x * Constants::mapSize) * 4 + 1].color = color;
                backAgent[(y + x * Constants::mapSize) * 4 + 2].color = color;
                backAgent[(y + x * Constants::mapSize) * 4 + 3].color = color;
            }
        }
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
        Window.draw(buffer.getFrontAgentMap());

        NormalizeViewport(Window, FieldOfView, sf::Vector2f(Fract(CameraPosition.x), Fract(CameraPosition.y)));

        IsGridDisplayed = FieldOfView > 7;

        if (IsGridDisplayed)
            DrawGrid(Window, FieldOfView);

        cursorSprite.setPosition((sf::Vector2f)cursorPos - sf::Vector2f(50.05,50.05) - CameraPosition);
        Window.draw(cursorSprite);

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

    if (Event.type == sf::Event::MouseButtonPressed) {
        sf::Vector2i pixelPos = sf::Mouse::getPosition(Window);
        sf::Vector2f worldPos = Window.mapPixelToCoords(pixelPos) + CameraPosition;

        cursorPos = sf::Vector2i((int)(worldPos.x + 50), (int)(worldPos.y + 50));
    }

    if (Event.type == sf::Event::KeyPressed) {
        if (Event.key.code == sf::Keyboard::O) SignalMapMode = !SignalMapMode;
        if (Event.key.code == sf::Keyboard::P) SimulationToRender.togglePause();
    }
}
