#pragma once
#include "ui.hpp"

void testByUI()
{
    std::string serverDir = "../Tools/UI/server.js";
    TEAM team = RED;
    int maxDepth = 20;
    int maxTime = 5;
    std::string fenCode = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1";

    // test situations
    // fenCode = "2BAKA1R1/9/4B4/P3P2cP/2P6/6P2/4p3p/9/2n1a4/3rk4/ w - - 0 1";
    // fenCode = "2b1ka3/4a4/2n1b4/p2cpRP1p/4P4/2B2N3/P4r2P/1CN6/4A4/4KAB2 w - - 0 1";
    // fenCode = "2bakabr1/9/4c1n2/p3p3p/1rpn2pc1/2PR5/P3P1P1P/C1N3N1C/9/2BAKABR1 w - - 0 1";
    // fenCode = "2bakabr1/6r2/1cn4cn/p3p3C/2p3P2/9/P1P1P3P/1CN3N2/9/R1BAKABR1 w - - 0 1";
    // fenCode = "rCbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/7C1/9/RNBAKABNR w - - 0 1";
    // fenCode = "9/3N1k3/9/1P3N2P/9/9/5r3/8B/3p5/3AK1B2 w - - 0 1"; // 绝杀搜索耗时过长
    // fenCode = "9/3k5/9/1P1N1c2P/4N4/9/9/8B/3p4r/3AK1B2 w - - 0 1";
    // fenCode = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1";
    // fenCode = "2baka1r1/3R5/c1n1b1cCn/p1p1p2R1/1r4p1p/1CP6/P3P1P1P/2N3N2/9/2BAKAB2 w - - 0 1"; // 节点数暴增
    // fenCode = "1nbaka1n1/r8/4b1c2/p1p3N2/4p3p/2P6/P3P3P/1CNCB4/4A4/R1BAK2c1 w - - 0 1";
    // fenCode = "4ka3/4aRP2/2n1b3b/p3p3p/9/4P4/P5r1P/4B4/c2CARN2/2rAK4 w - - 0 1";
    // fenCode = "2c1kab2/4aR3/4b4/2P1R1C1p/p5p2/8P/Pr2P1P2/4B1N2/1r7/c2AKAB2 w - - 0 1";
    // fenCode = "1rbakabnr/9/1cn4c1/p3p1p1p/2p6/9/P1P1P1P1P/4C1NC1/8R/RNBAKAB2 w - - 0 1";

    // fenCode = "r1bakabnr/9/1cn4c1/p3p1p1p/2p6/9/P1P1P1P1P/1CN4C1/R8/2BAKABNR w - - 0 1";

    ui(serverDir, team, maxDepth, maxTime, fenCode);
}
