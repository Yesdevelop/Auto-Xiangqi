#pragma once
#include "search.hpp"
#include "ui.hpp"

/// 带UI的测试
void ui(TEAM team, int maxDepth)
{
    int count = 0;
    PIECEID_MAP pieceidMap = DEFAULT_MAP;
    // pieceidMap = fenToPieceidMap("3aka3/1R1c1r3/2rcb1n2/p3p1C1p/2pn2bN1/1R7/P1P1P3P/2N3C2/9/2BAKAB2 w"); // 调试局面时使用
    Board board = Board(pieceidMap, RED);
    board.print();

    serverInit(board);

    Search s{};
    std::cout << "Open Chess98/UI/index.html to play chess\n" << std::endl;

    std::string moveFileContent = "____";
    PIECES eatens{};
    while (true)
    {
        if (board.team == team)
        {
            count++;
            std::cout << count;
            Result node = s.searchMain(board, maxDepth, 3);
            eatens.emplace_back(board.doMove(node.move));
            setBoardCode(board);
            board.print();

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
                        if (eatens.size() > 1)
                        {
                            FILE *file = nullptr;
                            errno_t result = fopen_s(&file, "./_move_.txt", "w+");
                            if (result == 0)
                            {
                                board.undoMove(board.historyMoves.back(), eatens.back());
                                eatens.pop_back();
                                board.undoMove(board.historyMoves.back(), eatens.back());
                                eatens.pop_back();
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
                        eatens.emplace_back(board.doMove(move));
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
