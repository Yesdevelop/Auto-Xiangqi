#pragma once
#include <iostream>
#include <string>
#include <cstdio>
#include <ctime>
#include "base.hpp"

std::string nnue_str = "[";
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

std::string getTimestampedFilename()
{
    std::time_t now = std::time(nullptr);
    std::tm *tm = std::localtime(&now);
    char buffer[20];
    std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", tm);
    return std::string(buffer);
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

std::string filename = getUniqueRandomFilename();
bool appExit = false;

void saveNNUE()
{
    nnue_str.pop_back();
    nnue_str = replaceAll(nnue_str, "}{", "},{");
    system("if not exist \"nnue\" mkdir nnue");
    writeFile("nnue/" + filename + ".json", nnue_str + "]");
}
