#pragma once

#define NNUE

#ifdef NNUE
#include "base.hpp"

// 这里放配置
const std::string NNUE_OUTPUT_DIR = "../nnue/data/"; // 首先你需要创建这个目录，才能写这个目录。后面要加尾随斜杠
const int NNUE_DEPTH = 7;                            // 最大搜索深度
const int NNUE_RANDOM_MOVE_COUNT = 5;                // 开局随机走法次数
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
    std::string filename = "";
    for (int i = 0; i < 16; i++)
    {
        std::mt19937 engine(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        std::uniform_int_distribution<size_t> distA(65, 90); // 大写字母
        std::uniform_int_distribution<size_t> distB(97, 122); // 小写字母
        std::uniform_int_distribution<size_t> distC(48, 57); // 数字
        int decision = std::uniform_int_distribution<size_t>(1, 3)(engine); // 决定
        if (decision == 1)
        {
            filename += distA(engine);
        }
        else if (decision == 2)
        {
            filename += distB(engine);
        }
        else
        {
            filename += distC(engine);
        }
    }
    return filename;
}

std::string NNUE_filecontent = "[";
bool NNUE_appexit = false;
const std::string NNUE_FILENAME = getUniqueRandomFilename();

void saveNNUE()
{
    if (NNUE_filecontent.back() == ',')
    {
        NNUE_filecontent.pop_back();
    }
    NNUE_filecontent = replaceAll(NNUE_filecontent, "}{", "},{");
    writeFile(NNUE_OUTPUT_DIR + NNUE_FILENAME + ".json", NNUE_filecontent + "]");
}
#endif
