#pragma once
#include "ui.hpp"

void testByUI()
{
    TEAM team = RED;
    bool aiFirst = true;
    int maxDepth = 8;
    int maxTime = 2;
    std::string fenCode = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1";

    // test situations

    // fenCode = "4k4/9/9/5R3/9/9/9/4p4/3p1p3/2p1K1p2 w"; // 长将局面
    // fenCode = "3k1a2n/4a4/6R2/9/9/9/9/B8/9/r1B1K4 w"; // 长捉局面 - 车捉马
    // fenCode = "r1b2k2c/1C7/3Nb4/p1p6/9/1C7/P1P6/9/9/4K3c w"; // 长捉局面 - 炮捉车
    // fenCode = "2raka3/2n6/4b4/9/9/4N4/9/3A5/4AK1R1/6B1c w"; // 长捉局面 - 车捉炮
    // fenCode = "2rak1c2/2n1a4/4b2R1/5N3/9/9/9/3A1K3/4A4/6B2 w"; // 有奇怪bug的局面
    // fenCode = "3akar2/9/4b4/9/6b2/9/P8/n3B4/2R1A4/3AK1B2 w"; // 长捉局面：马捉车
    // fenCode = "2baka3/9/c3b1n2/p3p3p/1nr6/1N1r2P2/P3P3P/3CC2R1/9/1RBAKAB2 w - - 0 1"; // 走出奇怪着法的局面
    // fenCode = "2bakab2/9/2R3n2/p1p1p1p1p/3r5/2P6/P1r1c1PcP/C3C3N/4N4/2BAKAB1R w - - 0 1"; // 智障局面
    // fenCode = "3ckab2/2r1a4/2n1bc3/1RN1p3p/P5p2/9/3n2P1P/C3C3N/4A4/2B1KAB2 w - - 0 1"; // 一个不是很显眼的漏招
    // fenCode = "3k1ab2/2n1a4/4b4/1cr1C2RN/p8/4N4/4P1P1P/4B4/4A4/2BAK4 w - - 0 1"; // 大漏
    // fenCode = "3ak4/4a4/4b2c1/p3R1p1p/9/4C4/P1P1P1P1P/2N1Br3/3cNr3/2BAKA1R1 w - - 0 1"; // 大漏
    // fenCode = "1rbakab2/8r/2n3nc1/p3p1p1p/2p6/6P2/PcP1P3P/2N1C1NC1/1R7/2BAKAB1R w - - 0 1"; // 进兵
    // fenCode = "C2a5/5k1P1/3a5/9/9/6r2/9/9/1cp1p3C/4KA3 w - - 0 1"; // 杀棋送将
    // fenCode = "2bak1C2/4a1R2/c6c1/p3p1p2/1r6p/9/P3P1P1P/6NR1/3r5/3AKAB2 w - - 0 1"; // 深度不够
    // fenCode = "2bak4/3Ra4/3n5/p8/2b2PP1p/2NR5/P1r1N3P/1r2n4/4A4/2BK1A3 w - - 0 1" // 深度不够
    // fenCode = "4k4/9/9/9/9/9/9/9/9/4K4 w"; // 测试对面笑的局面

    ui(team, aiFirst, maxDepth, maxTime, fenCode);
}

#ifdef NNUE
void testGenerateNNUE()
{
    int randomDepth = NNUE_RANDOM_MOVE_COUNT;
    int maxDepth = NNUE_DEPTH;
#ifdef _WIN32
    std::string exePath = NNUE_RESTART_EXE_FILE;
#elif __unix__
    std::string exePath = NNUE_RESTART_LINUX_FILE;
#endif

    const std::string fenCode = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1";
    Search s{fenToPieceidMap(fenCode), RED};
    int count = 0;

    try
    {
        // 前几步随机
        for (int i = 0; i < randomDepth; i++)
        {
            count++;
            std::cout << count << "---------------" << std::endl;
            s.searchGenereateNNUE(maxDepth, 3);
            Move m = getRandomFromVector<Move>(s.rootMoves);
            s.board.doMove(m);
            saveNNUE();
        }
        while (s.board.historyMoves.size() < 140 && NNUE_appexit == false)
        {
            count++;
            std::cout << count << "---------------" << std::endl;
            Result a = s.searchGenereateNNUE(maxDepth, 3);
            if (a.move.id != -1)
            {
                s.board.doMove(a.move);
            }
            saveNNUE();
        }
    }
    catch (std::exception err)
    {
        std::cout << "A fucking error occured" << std::endl;
        std::cerr << err.what() << std::endl;
        system_("pause");
    }
#ifdef _WIN32
    system_("start /min " + exePath);
#elif __unix__
    system_("nohup " + exePath + " &");
#endif
}
#endif
