#pragma once
#include "search.hpp"
#include "ui.hpp"

/// 带UI的测试
void ui(TEAM team, int maxDepth)
{
    int count = 0;
    PIECEID_MAP pieceidMap = DEFAULT_MAP;
    pieceidMap = fenToPieceidMap("4kab2/4a4/4b3c/pr2C3R/4P4/2c6/P1n3r1P/C8/4N4/R1BAKAB2 w - - 0 1"); // 调试局面时使用
    //pieceidMap = fenToPieceidMap("3k1ab2/4a4/4b3c/p3CR3/4P4/4c4/P2r4P/4C4/4N4/2BAKAB2 w - - 0 1");
    //pieceidMap = fenToPieceidMap("1R1rkab2/4a4/4b3c/p3CR3/4P4/4c4/P2r4P/4C4/4N4/2BAKAB2 w - - 0 1");
    Board board = Board(pieceidMap, RED);
	std::cout << board.evaluate() << std::endl;
    serverInit(board);

    Search s{};
    std::cout << "Open Chess98/UI/index.html to play chess\n" << std::endl;

    std::string moveFileContent = "____";
    while (true)
    {
        if (board.team == team)
        {
            count++;
            std::cout << count;
            Result node = s.searchMain(board, maxDepth, 3);
            board.doMove(node.move);
            setBoardCode(board);

            FILE *file = nullptr;
            errno_t result = fopen_s(&file, "./_move_.txt", "r");
            if (result == 0)
            {
                char buffer[4]{};
                fseek(file, 0, SEEK_SET);
                fread(&buffer, 4, 1, file);
                fclose(file);
                moveFileContent = std::string(buffer).substr(0, 4);
            }
            else
                std::cerr << "CANNOT OPEN FILE!" << result << std::endl;
        }
        else
        {
            while (true)
            {
                FILE *file = nullptr;
                errno_t result = fopen_s(&file, "./_move_.txt", "r");
                if (result == 0)
                {
                    char buffer[4]{};
                    fseek(file, 0, SEEK_SET);
                    fread(&buffer, 4, 1, file);
                    fclose(file);
                    std::string content = std::string(buffer).substr(0, 4);
                    if (content == "undo")
                    {
                        if (board.historyEatens.size() > 1)
                        {
                            FILE *file = nullptr;
                            errno_t result = fopen_s(&file, "./_move_.txt", "w+");
                            if (result == 0)
                            {
                                board.undoMove(board.historyMoves.back(), board.historyEatens.back());
                                board.undoMove(board.historyMoves.back(), board.historyEatens.back());
                                setBoardCode(board);
                                moveFileContent = "____";
                                fwrite("____", 4, 1, file);
                                fclose(file);
                                count--;
                                std::cout << "undo" << std::endl;
                            }
                        }
                        break;
                    }
                    if (content != moveFileContent)
                    {
                        moveFileContent = content;
                        int x1 = std::stoi(content.substr(0, 1));
                        int y1 = std::stoi(content.substr(1, 1));
                        int x2 = std::stoi(content.substr(2, 1));
                        int y2 = std::stoi(content.substr(3, 1));
                        Move move{x1, y1, x2, y2};
                        board.doMove(move);
                        break;
                    }
                    wait(50);
                }
                else
                    std::cerr << "CANNOT OPEN FILE!" << result << std::endl;
            }
        }
    }
}
