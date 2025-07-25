#pragma once
#include "ui.hpp"

void testByUI()
{
    std::string serverDir = "../Tools/UI/server.js";
    TEAM team = RED;
    bool aiFirst = true;
    int maxDepth = 20;
    int maxTime = 3;
    std::string fenCode = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1";

    // test situations

    // fenCode = "4k4/9/9/5R3/9/9/9/4p4/3p1p3/2p1K1p2 w"; // 长将局面
    // fenCode = "3k1a2n/4a4/6R2/9/9/9/9/B8/9/r1B1K4 w"; // 长捉局面 - 车捉马
    // fenCode = "r1b2k2c/1C7/3Nb4/p1p6/9/1C7/P1P6/9/9/4K3c w"; // 长捉局面 - 炮捉车
    // fenCode = "2raka3/2n6/4b4/9/9/4N4/9/3A5/4AK1R1/6B1c w"; // 长捉局面 - 车捉炮
    // fenCode = "2rak1c2/2n1a4/4b2R1/5N3/9/9/9/3A1K3/4A4/6B2 w"; // 有奇怪bug的局面
    // fenCode = "3akar2/9/4b4/9/6b2/9/P8/n3B4/2R1A4/3AK1B2 w"; // 长捉局面：马捉车

    ui(serverDir, team, aiFirst, maxDepth, maxTime, fenCode);
}
