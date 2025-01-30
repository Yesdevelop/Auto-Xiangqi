#pragma once
#include "base.hpp"

using BITLINE = unsigned;
using BITARRAY_X = std::array<BITLINE, 9>;
using BITARRAY_Y = std::array<BITLINE, 10>;
using REGION_ROOK = std::array<int, 2>; // 车的着法起始点终结点
using REGION_CANNON = std::array<int, 4>; // 炮的着法起始点终结点
using TYPE_ROOK_CACHE = std::map<int, std::map<BITLINE, REGION_ROOK>>;
using TYPE_CANNON_CACHE = std::map<int, std::map<BITLINE, REGION_CANNON>>;

/// @brief 位棋盘(车、炮着法生成)
class BitBoard
{
public:
    BitBoard(PIECEID_MAP pieceidMap)
    {
        // 初始化棋盘
        for (int x = 0; x < 9; x++)
        {
            for (int y = 0; y < 10; y++)
            {
                if (pieceidMap[x][y] != EMPTY_PIECEID)
                    this->setBit(x, y);
            }
        }
        // 初始化车的着法缓存
        for (BITLINE i = 1; i <= pow(2, 10); i++)
        {

        }
    }

    TYPE_ROOK_CACHE rookCache{};
    TYPE_CANNON_CACHE cannonCache{};

    BITARRAY_X xBitBoard{0, 0, 0, 0, 0, 0, 0, 0, 0};
    BITARRAY_Y yBitBoard{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

private:
    /// @brief 获取bitline二进制的第n位
    /// @param bitline
    /// @param index
    /// @return
    int getBit(BITLINE bitline, int index)
    {
        return (bitline >> index) & 1;
    }

    /// @brief 设置棋盘上x, y的数为1
    /// @param x
    /// @param y
    void setBit(int x, int y)
    {
        this->xBitBoard[x] |= (1 << y);
        this->yBitBoard[y] |= (1 << x);
    }

    /// @brief 设置棋盘上x, y的数为0
    /// @param x
    /// @param y
    void deleteBit(int x, int y)
    {
        this->xBitBoard[x] &= ~(1 << y);
        this->yBitBoard[y] &= ~(1 << x);
    }

    /// @brief 获取车的着法范围
    /// @param bitline 没有合法性检查
    /// @param num 没有合法性检查
    /// @return
    REGION_ROOK generateRookRegion(BITLINE bitline, int num)
    {
        
    }

    /// @brief 获取炮的着法范围
    /// @param bitline
    /// @param num
    /// @return
    REGION_CANNON generateCannonRegion(BITLINE bitline, int num)
    {
        return REGION_CANNON{};
    }
};
