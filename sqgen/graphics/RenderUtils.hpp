#pragma once

void NormalizeViewport(sf::RenderWindow& Window, float fov, sf::Vector2f CameraPosition);

void DrawLine(sf::RenderWindow& Window, float x1, float y1, float x2, float y2, sf::Color Color);

void DrawHollowSquare(sf::RenderWindow& Window, float x, float y, sf::Color Color);

int XUnitsCount(sf::RenderWindow& Window, float fov);

int YUnitsCount(sf::RenderWindow& Window, float fov);

float Fract(float f);

float Clamp(float v, float l, float h);

void DrawGrid(sf::RenderWindow& Window, float fov);

void genVertexArray(sf::VertexArray& arr);

sf::Color randomAgentColor();

sf::Color mutateColor(sf::Color color);