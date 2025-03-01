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
    // PIECEID_MAP pieceidMap = *fenToPieceidMap("3akab2/9/4b1nr1/p5R1p/4p4/2Ncn1P2/PR2Nr2P/2C6/4A4/2B1KAB2 w - - 0 1");
    Board board = Board(DEFAULT_MAP, RED);
    board.print();

    serverInit(board);

    Search s;
    s.searchInit(board);
    system("cd ../ChessUI && index.html");

    std::string moveFileContent = "____";
    while (true)
    {
        if (board.team == team)
        {
            Node node = s.searchMain(board, maxDepth, 3);
            board.doMove(node.move);
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
                board = Board(board.pieceidMap, board.team);
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
