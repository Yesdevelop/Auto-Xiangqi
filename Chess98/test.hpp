#pragma once
#include "search.hpp"
#include "ui.hpp"

void testWithUI(TEAM team, int maxDepth);
void checkingTest();

void test(TEAM team = BLACK, int maxDepth = 10)
{
    testWithUI(team, maxDepth);
}

/// 测试将军检测函数
void checkingTest()
{
    PIECEID_MAP MAP{
        {{R_ROOK, 0, 0, R_PAWN, 0, 0, B_PAWN, 0, 0, B_ROOK},
         {B_CANNON, 0, R_CANNON, 0, 0, 0, 0, B_CANNON, 0, B_KNIGHT},
         {R_BISHOP, 0, 0, R_PAWN, 0, 0, B_PAWN, 0, 0, B_BISHOP},
         {R_GUARD, 0, 0, 0, 0, 0, 0, 0, 0, B_GUARD},
         {R_KING, 0, 0, 0, R_CANNON, 0, 0, 0, 0, B_KING},
         {R_GUARD, 0, 0, 0, 0, 0, 0, 0, 0, B_GUARD},
         {R_BISHOP, 0, 0, R_PAWN, 0, 0, B_PAWN, 0, 0, B_BISHOP},
         {R_KNIGHT, 0, R_CANNON, 0, 0, 0, 0, B_CANNON, 0, B_KNIGHT},
         {R_ROOK, 0, 0, R_PAWN, 0, 0, B_PAWN, 0, 0, B_ROOK}}};
    Board board = Board(MAP, RED);
    bool s = inCheck(board);
    std::cout << s << std::endl;
}

/// 带UI的测试
void testWithUI(TEAM team, int maxDepth)
{
    Board board = Board({{
        {0, 0, 0, R_PAWN, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, B_KNIGHT, 0, 0, 0, 0, 0},
        {0, 0, 0, B_ROOK, R_CANNON, 0, 0, 0, 0, 0},
        {0, R_GUARD, R_KING, R_KNIGHT, R_ROOK, 0, 0, 0, B_GUARD, B_KING},
        {R_GUARD, 0, 0, 0, 0, B_ROOK, 0, 0, 0, B_GUARD},
        {0, 0, 0, 0, R_BISHOP, 0, 0, R_PAWN, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    }}, team);
    board.print();

    serverInit(board);

    Search s;
    s.searchInit(board);

    std::string moveFileContent;
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
