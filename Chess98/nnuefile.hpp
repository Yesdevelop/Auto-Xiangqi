#pragma once
#include <iostream>
#include <string>
#include <cstdio>
#include <ctime>
#include "base.hpp"

// 这里放配置
const std::string NNUE_OUTPUT_DIR = "../nnue/data/"; // 首先你需要创建这个目录，才能写这个目录。后面要加尾随斜杠
const int NNUE_DEPTH = 7; // 最大搜索深度
const int NNUE_RANDOM_MOVE_COUNT = 5; // 开局随机走法次数
const std::string NNUE_RESTART_EXE_FILE = "./a.exe"; // 跑完一局继续跑的exe文件路径

std::string replaceAll(std::string resource_str, std::string sub_str, std::string new_str)
{
    std::string dst_str = resource_str;
    std::string::size_type pos = 0;
    while ((pos = dst_str.find(sub_str)) != std::string::npos) // 替换所有指定子串
    {
        dst_str.replace(pos, sub_str.length(), new_str);
    }
    return dst_str;
}

std::string getUniqueRandomFilename()
{
    int counter = 0;
    std::string filename;
    bool exists;
    do
    {
        auto now = std::chrono::high_resolution_clock::now();

        // 转换为纳秒级时间戳（int64_t）
        auto timestamp = std::chrono::time_point_cast<std::chrono::nanoseconds>(now)
                             .time_since_epoch()
                             .count();
        filename = std::to_string(timestamp) + "_" +
                   std::to_string(counter++) + "_" +
                   std::to_string(std::rand());

        // Check if file exists using system command
        std::string cmd = "dir /b nnue\\" + filename + ".json > nul 2>&1";
        exists = system(cmd.c_str()) == 0;
    } while (exists);
    return filename;
}

std::string NNUE_filecontent = "[";
bool NNUE_appexit = false;
const std::string NNUE_FILENAME = getUniqueRandomFilename();

void saveNNUE()
{
    NNUE_filecontent.pop_back();
    NNUE_filecontent = replaceAll(NNUE_filecontent, "}{", "},{");
    writeFile(NNUE_OUTPUT_DIR + NNUE_FILENAME + ".json", NNUE_filecontent + "]");
}
