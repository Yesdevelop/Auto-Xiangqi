/**
 * 说明见 /DEV.md
 */

#pragma once
#include "base.hpp"

using BITLINE = unsigned;
using BITARRAY_X = std::array<BITLINE, 9>;
using BITARRAY_Y = std::array<BITLINE, 10>;
using REGION_ROOK = std::array<int, 2>;
using REGION_CANNON = std::array<int, 4>;
using TYPE_ROOK_CACHE = std::array<std::array<REGION_ROOK, 10>, 1024>;
using TYPE_CANNON_CACHE = std::array<std::array<REGION_CANNON, 10>, 1024>;

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
        // 初始化车、炮的着法缓存
        for (BITLINE bitline = 1; bitline <= pow(2, 10); bitline++)
        {
            for (int index = 0; index < 10; index++)
            {
                if (this->getBit(bitline, index) == 1)
                {
                    this->rookCache[bitline][index] = this->generateRookRegion(bitline, index);
                    this->cannonCache[bitline][index] = this->generateCannonRegion(bitline, index);
                }
            }
        }
    }

    /// @brief 获取车的着法缓存
    /// @param bitline
    /// @param index
    /// @param endpos 只接受8或9
    /// @return
    REGION_ROOK getRookRegion(BITLINE bitline, int index, int endpos)
    {
        REGION_ROOK result = this->rookCache[bitline][index];
        if (endpos == 8)
            return result;
        if (result[1] == 8 && this->getBit(bitline, 9) == 0 && (this->getBit(bitline, 8) != 1 || index == 8))
            result[1] = 9;
        if (index == 9)
            result[1] = 9;
        return result;
    }

    /// @brief 获取炮的着法缓存
    /// @param bitline
    /// @param index
    /// @param endpos
    /// @return
    REGION_CANNON getCannonRegion(BITLINE bitline, int index, int endpos)
    {
        REGION_CANNON result = this->cannonCache[bitline][index];
        if (endpos == 8)
            return result;
        if (result[2] == 8 && this->getBit(bitline, 9) == 0 && (this->getBit(bitline, 8) != 1 || index == 8))
            result[2] = result[3] = 9;
        if (index == 9)
            result[2] = result[3] = 9;
        return result;
    }

    TYPE_ROOK_CACHE rookCache{};
    TYPE_CANNON_CACHE cannonCache{};

    BITARRAY_X xBitBoard{0, 0, 0, 0, 0, 0, 0, 0, 0};
    BITARRAY_Y yBitBoard{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

private:
    friend class Board;

    /// @brief 获取bitline二进制的第n位(从右边到左边)
    /// @param bitline
    /// @param index
    /// @return
    int getBit(BITLINE bitline, int index) const
    {
        return (bitline >> index) & 1;
    }
    /// @brief 步进
    /// @param x1
    /// @param y1
    /// @param x2
    /// @param y2
    /// @return 是否有吃子
    void doMove(int x1, int y1, int x2, int y2)
    {
        bool ret = bool(this->getBit(x2, y2));
        this->deleteBit(x1, y1);
        this->setBit(x2, y2);
    }

    /// @brief 撤销步进
    /// @param x1
    /// @param y1
    /// @param x2
    /// @param y2
    /// @param eaten
    void undoMove(int x1, int y1, int x2, int y2, bool eaten)
    {
        this->setBit(x1, y1);
        if (!eaten)
        {
            this->deleteBit(x2, y2);
        }
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
    /// @param bitline
    /// @param index 车的所在位置
    /// @param end 截止点，接受8或9
    /// @return
    REGION_ROOK generateRookRegion(BITLINE bitline, int index) const
    {
        int beg = 0;
        int end = 8;
        for (int pos = index - 1; pos >= 0; pos--)
        {
            if (this->getBit(bitline, pos) != 0)
            {
                beg = pos;
                break;
            }
        }
        for (int pos = index + 1; pos <= 8; pos++)
        {
            if (this->getBit(bitline, pos) != 0)
            {
                end = pos;
                break;
            }
        }

        return REGION_ROOK{beg, end};
    }

    /// @brief 获取炮的着法范围
    /// @param bitline
    /// @param index 炮的所在位置
    /// @param end 截止点，接受9或10
    /// @return
    REGION_CANNON generateCannonRegion(BITLINE bitline, int index) const
    {
        int eaten1 = 0;
        int beg = 0;
        int end = 8;
        int eaten2 = 8;
        for (int pos = index - 1; pos >= 0; pos--)
        {
            if (this->getBit(bitline, pos) != 0)
            {
                beg = pos + 1;
                eaten1 = pos + 1;
                for (int pos2 = pos - 1; pos2 >= 0; pos2--)
                {
                    if (this->getBit(bitline, pos2) != 0)
                    {
                        eaten1 = pos2;
                        break;
                    }
                }
                break;
            }
        }
        for (int pos = index + 1; pos <= 8; pos++)
        {
            if (this->getBit(bitline, pos) != 0)
            {
                end = pos - 1;
                eaten2 = pos - 1;
                for (int pos2 = pos + 1; pos2 <= 9; pos2++)
                {
                    if (this->getBit(bitline, pos2) != 0)
                    {
                        eaten2 = pos2;
                        break;
                    }
                }
                break;
            }
        }

        return REGION_CANNON{eaten1, beg, end, eaten2};
    }
};
