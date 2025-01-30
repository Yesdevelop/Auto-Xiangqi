#include "test.hpp"
#include <iostream>
#include <random>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <io.h>
#include <omp.h>

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

void traverse() {
    std::string rootPath = "../need_to_estimate";
    std::vector<std::string> filePaths;
    getFiles(rootPath, filePaths);
    std::cout << filePaths.size() << std::endl;
    for (int cnt = 0;cnt < filePaths.size();cnt++) {
        std::string path = filePaths[cnt];
        // init status
        Board board = Board(DEFAULT_MAP, RED);
        Search s;
        s.searchInit(board,16);
        // prepare to read recoards
        std::ifstream in(path);
        std::string moveStr;
        std::vector<std::string> dumps;
        while (getline(in, moveStr)) {
            // get move from buffer
            const int combinatePos = atoi(moveStr.c_str());
            std::string mv = std::to_string(combinatePos);
            // estimate the game status
            Node BestNode = s.searchMain(board, 6, 1);
            // recoard the move with assessment result
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
        std::string filename = std::to_string(cnt) + ".txt";
        std::string filepath = "../assessment_result/" + filename;
        std::ofstream outfile;
        outfile.open(filepath);
        for (std::string& output : dumps) {
            outfile << output << std::endl;
        }
        outfile.close();
    }
}


int main()
{
    traverse();
    return 0;
}