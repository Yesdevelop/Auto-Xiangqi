#include <iostream>
#include <random>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <io.h>
#include <chrono>
#include <iomanip>
#include <sstream>
#include "test.hpp"

/* ***** 跑谱器 ***** */

/// @brief 根据Python脚本生成的简化棋谱计算评估值，从而训练叶节点估值网络

void getFiles(std::string path, std::vector<std::string>& files)
{
    intptr_t hFile = 0;
    struct _finddata_t fileinfo {};
    std::string p;
    if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
    {
        do
        {
            if ((fileinfo.attrib & _A_SUBDIR))
            {
                if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
                    getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
            }
            else
            {
                files.push_back(p.assign(path).append("\\").append(fileinfo.name));
            }
        } while (_findnext(hFile, &fileinfo) == 0);
        _findclose(hFile);
    }
}

// 检查文件是否存在的函数
bool fileExists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

// 生成基于当前日期和时间以及随机数的唯一文件名的函数
std::string generateUniqueFilename(const std::string& outputPath) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 9999);

    std::string filename;
    do {
        // 获取当前时间
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;

        // 生成随机数
        int random_number = dis(gen);

        // 组合时间戳和随机数生成文件名
        std::stringstream ss;
        ss << outputPath << std::put_time(std::localtime(&in_time_t), "%Y%m%d%H%M%S") << millis << "_" << random_number << ".txt";
        filename = ss.str();

    } while (fileExists(filename)); // 如果文件已经存在，重新生成

    return filename;
}

void traverse() {
    std::string rootPath = "../need_to_estimate";
    std::vector<std::string> filePaths;
    getFiles(rootPath, filePaths);
    std::cout << filePaths.size() << std::endl;
    for (int cnt = 0; cnt < filePaths.size(); cnt++) {
        std::string path = filePaths[cnt];
        // init status
        Board board = Board(DEFAULT_MAP, RED);
        Search s;
        s.searchInit(board, 16);
        // prepare to read records
        std::ifstream in(path);
        std::string moveStr;
        std::vector<std::string> dumps;
        while (getline(in, moveStr)) {
            // get move from buffer
            const int combinatePos = atoi(moveStr.c_str());
            std::string mv = std::to_string(combinatePos);
            // estimate the game status
            Node BestNode = s.searchMain(board, 6, 1);
            // record the move with assessment result
            std::string output = mv + " " + std::to_string(BestNode.score);
            dumps.push_back(output);
            // next step
            int src = combinatePos & 255;
            int dst = combinatePos >> 8;
            int xSrc = (src & 15) - 3;
            int ySrc = 12 - (src >> 4);
            int xDst = (dst & 15) - 3;
            int yDst = 12 - (dst >> 4);
            Move tMove = Move(xSrc, ySrc, xDst, yDst);
            board.doMove(tMove);
        }
        std::string filepath = generateUniqueFilename("../assessment_result/");
        std::ofstream outfile;
        outfile.open(filepath);
        for (std::string& output : dumps) {
            outfile << output << std::endl;
        }
        outfile.close();
    }
}

void randomTraverse() {
    std::string outputPath = "../dump/";
    for (int i = 0; i < 1000000; i++) {
        Board board = Board(DEFAULT_MAP, RED);
        Search s;
        s.searchInit(board, 20);
        std::vector<std::string> dumps;
        std::srand(time(0));
        for (int a = 0; a < 200; a++) {
            int vlRandom = rand() % 100;
            std::string mv;
            std::string vl;
            std::string flag;
            Move tMove;
            if (vlRandom <= 20) {
                MOVES mvs = Moves::getMoves(board);
                MOVES valid_mvs;
                for (const Move& move : mvs) {
                    Piece eaten = board.doMove(move);
                    board.team = -board.team;
                    bool bCheck = inCheck(board);
                    board.team = -board.team;
                    board.undoMove(move, eaten);
                    if (!bCheck) {
                        valid_mvs.emplace_back(move);
                    }
                }
                if (valid_mvs.empty()) {
                    break;
                }
                else {
                    tMove = valid_mvs[rand() % (valid_mvs.size())];
                    int src = (tMove.x1 + 3) + ((tMove.y1 + 3) << 4);
                    int dst = (tMove.x2 + 3) + ((tMove.y2 + 3) << 4);
                    mv = std::to_string(src + (dst << 8));
                    vl = std::to_string(0);
                    flag = "Random";
                }
            }
            else {
                Node BestNode = s.searchMain(board, 6, 1);
                tMove = BestNode.move;
                int src = (tMove.x1 + 3) + ((tMove.y1 + 3) << 4);
                int dst = (tMove.x2 + 3) + ((tMove.y2 + 3) << 4);
                mv = std::to_string(src + (dst << 8));
                vl = std::to_string(BestNode.score);
                flag = "Okay";
            }
            std::string output = mv + " " + vl + " " + flag;
            dumps.emplace_back(output);
            board.doMove(tMove);
            // 检查是否结束游戏
            if (board.isKingLive(RED) == false || board.isKingLive(BLACK) == false) {
                break;
            }
        }
        std::ofstream outfile;
        std::string filepath = generateUniqueFilename(outputPath);
        outfile.open(filepath);
        for (std::string& output : dumps) {
            outfile << output << std::endl;
        }
        outfile.close();
    }
}

int main()
{
    randomTraverse();
    return 0;
}