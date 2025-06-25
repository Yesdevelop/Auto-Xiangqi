#pragma once
#include "ui.hpp"

void testByUI()
{
    std::string serverDir = "../Tools/UI/server.js";
    TEAM team = RED;
    int maxDepth = 15;
    int maxTime = 3;
    std::string fenCode = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1";

    // test situations
    // fenCode = "2BAKA1R1/9/4B4/P3P2cP/2P6/6P2/4p3p/9/2n1a4/3rk4/ w - - 0 1";
    // fenCode = "2b1ka3/4a4/2n1b4/p2cpRP1p/4P4/2B2N3/P4r2P/1CN6/4A4/4KAB2 w - - 0 1";

    ui(serverDir, team, maxDepth, maxTime, fenCode);
}
