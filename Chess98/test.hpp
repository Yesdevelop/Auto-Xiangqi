#pragma once
#include "search.hpp"
#include "ui.hpp"

void testWithUI(TEAM team, int maxDepth);

void test(TEAM team = BLACK, int maxDepth = 16)
{
    testWithUI(team, maxDepth);
}


/// 带UI的测试
void testWithUI(TEAM team, int maxDepth)
{
    int count = 0;
    PIECEID_MAP pieceidMap = DEFAULT_MAP;
    // pieceidMap = fenToPieceidMap("3akabr1/5R3/2n1b2c1/p6Rp/4p1p2/9/P1p2Nn1P/4C1N2/9/2rAKAB2 w - - 0 1"); // 调试局面时使用
    Board board = Board(pieceidMap, RED);
    board.print();

    serverInit(board);

    Search s;
    system("cd ../UI && index.html");

    std::string moveFileContent = "____";
    std::vector<Piece> eatens{};
    while (true)
    {
        if (board.team == team)
        {
            count++;
            std::cout << count;
            Result node = s.searchMain(board, maxDepth, 3);
            eatens.emplace_back(board.doMove(node.move));
            setBoardCode(board);
            MOVES _ = Moves::getMoves(board);

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
            {
                std::cerr << "CANNOT OPEN FILE!" << result << std::endl;
            }
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
                    Sleep(50);
                }
                else
                {
                    std::cerr << "CANNOT OPEN FILE!" << result << std::endl;
                }
            }
        }
    }
}
