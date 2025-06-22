#pragma once
#include "search.hpp"
#include "ui.hpp"

void ui(TEAM team, int maxDepth)
{
    int count = 0;
    PIECEID_MAP pieceidMap = DEFAULT_MAP;
    //pieceidMap = fenToPieceidMap("4k4/9/9/5R3/9/9/9/4p4/3p1p3/2p1K1p2 w - - 0 1"); // 调试局面时使用
    Board board = Board(pieceidMap, RED);
    board.print();
	std::thread serverInitThread(serverInit, board);
	serverInitThread.detach();

    Search s{};
    std::cout << "Open Chess98/UI/index.html to play chess\n"
              << std::endl;
    std::string moveFileContent = "____";

    while (true)
    {
        if (board.team == team)
        {
            count++;
            std::cout << count;
            Result node = s.searchMain(board, 13, 3);
            board.doMove(node.move);

            if (inCheck(board) == true)
                board.historyMoves.back().isCheckingMove = true;
            if (board.isRepeatStatus())
            {
                std::cout << "REPEAT STATUS!" << std::endl;
			}

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
