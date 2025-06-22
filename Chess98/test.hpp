#pragma once
#include "ui.hpp"

void testByUI()
{
    std::string serverDir = "../Tests/UI/server.js";
    TEAM team = RED;
    int maxDepth = 20;
    int maxTime = 3;
    std::string fenCode = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1";

    ui(serverDir, team, maxDepth, maxTime, fenCode);
}
