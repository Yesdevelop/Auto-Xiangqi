#pragma once
#include "ui.hpp"

void testByUI()
{
    std::string serverDir = "../Tools/UI/server.js";
    TEAM team = RED;
    int maxDepth = 15;
    int maxTime = 3;
    std::string fenCode = "2bak4/9/4ba3/8p/4P2N1/9/4r1P1P/3A5/3K3n1/R1C6 w - - 0 1";
    fenCode = "2ba1k3/9/3Nba3/4P3p/9/9/6P1P/3A1n3/2r6/R1CK5 w - - 0 1";
    // fenCode = "4Ck3/5a3/5ba2/5P3/9/9/7P1/3A2n2/2r6/R2K5 w";
     //fenCode = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1";
    // fenCode = "2bak4/9/4ba3/5N2p/4P4/9/2r3P1P/3A1n3/9/R1CK5 w - - 0 1";
     fenCode = "2ba5/5k3/3rb4/8p/9/9/6P1P/2n6/9/R3K4 w";
     fenCode = "4k4/9/9/5R3/9/9/9/4p4/3p1p3/2p1K1p2 w - - 0 1";

    ui(serverDir, team, maxDepth, maxTime, fenCode);
}

// # 调试局面大赏
// 
// fenCode = "2ba1k3/9/3Nba3/8p/4P4/9/2r3P1P/3A1n3/9/R1CK5 w - - 0 1";
// fenCode = "2bak4/9/4ba3/8p/4P2N1/9/4r1P1P/3A5/3K3n1/R1C6 w - - 0 1";
// fenCode = "2bak4/9/4ba3/5N2p/4P4/9/2r3P1P/3A1n3/9/R1CK5 w - - 0 1";
// fenCode = "2ba5/5k3/3rb4/8p/9/9/6P1P/2n6/9/R3K4 w";
