#pragma once
#include "search.hpp"
#include "ui.hpp"

void testWithUI(TEAM team, int maxDepth);

void test(TEAM team = BLACK, int maxDepth = 16)
{
    testWithUI(team, maxDepth);
}

void testOpenBook()
{
    Board board = Board(DEFAULT_MAP, RED);
    board.print();

    Search s;
    s.searchInit(board);

    Move bookMove = s.searchOpenBook(board);

    std::cout << (int)board.hashKey << " " << (int)board.hashLock << std::endl;
}

/// 带UI的测试
void testWithUI(TEAM team, int maxDepth)
{
    int count = 0;
    PIECEID_MAP pieceidMap = DEFAULT_MAP;
    // pieceidMap = fenToPieceidMap("2P1k1P2/3P1P3/9/4P4/9/2r6/9/9/9/4K4 w - - 0 1"); // 调试局面时使用
    Board board = Board(pieceidMap, RED);
    board.print();

    serverInit(board);

    Search s;
    s.searchInit(board);
    system("cd ../UI && index.html");

    std::string moveFileContent = "____";
    std::vector<Piece> eatens{};
    while (true)
    {
        if (board.team == team)
        {
            count++;
            std::cout << count;
            Root node = s.searchMain(board, maxDepth, 3);
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
                }
                else
                {
                    std::cerr << "CANNOT OPEN FILE!" << result << std::endl;
                }
                Sleep(400);
            }
        }
    }
}
