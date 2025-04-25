# Chess98

以下是关于代码细节的一些说明，注释里可能确实讲不清楚思路，在这里提供一下我设计的想法

## base.hpp

定义了代码所用到的常量和基本类，有以下这些：

- INF, BAN, ILLEGAL_VAL, U64
- PIECE_INDEX 整数类型，棋盘初始化时会生成棋子列表 Board.pieces，每一个 PIECE_INDEX 类型就相当于一个对特定 Piece 的引用，记录着这个 Piece 在 Board::pieces 里的索引号，若棋子不存在则为 EMPTY_INDEX
- PIECEID 整数类型，一个棋子的类别，比如黑车: -5，红帅: 1，不存在则为EMPTY_PIECEID，越界则为 OVERFLOW_PIECEID
- TEAM 整数类型，队伍类别，和 PIECEID 情况大致相同
- PIECEID_MAP 一个 9x10 格式的数组，代表中国象棋棋盘
- PIECE_NAME_PAIRS, NAME_PIECE_PAIRS

- Piece 类
    - team()
    - pieceid
    - x
    - y
    - pieceIndex 想要通过index访问就用Board::pieceIndex(myIndex)
    - isLive: bool
- Move 类
    - id: int 快速比较着法
    - val
    - moveType
    - ==, !=
- Result 类
    - val
    - move

## bitboard.hpp

内部实现细节很难说，我已经把它和 Board 类绑了 friend，毕竟这俩确实联系蛮紧密的，然后把其他方法全都放在了 private 里。可能用到的有这几个：

- getCannonRegion
- getRookRegion
- Board::getBitlineX
- Board::getBitlineY

这是着法生成的例子：

```cpp
MOVES Moves::rook(TEAM team, Board &board, int x, int y) // 车的着法生成
{
    MOVES result{};
    result.reserve(64);

    // 纵向着法
    BITLINE bitlineX = board.getBitLineX(x); // 纵向着法就是棋盘上长度为10的那一列
    REGION_ROOK regionX = board.bitboard->getRookRegion(bitlineX, y, 9); // 最后一个参数放9是告诉这个函数要获取0~9的REGION，获取一列10个子就代9进去
    for (int y2 = y + 1; y2 < regionX[1]; y2++)
        result.emplace_back(Move{x, y, x, y2});
    if (board.teamOn(x, regionX[1]) != team)
        result.emplace_back(Move{x, y, x, regionX[1]});
    for (int y2 = y - 1; y2 > regionX[0]; y2--)
        result.emplace_back(Move{x, y, x, y2});
    if (board.teamOn(x, regionX[0]) != team)
        result.emplace_back(Move{x, y, x, regionX[0]});

    // 横向着法
    BITLINE bitlineY = board.getBitLineY(y); // 横向着法就是长度为9的那一行
    REGION_ROOK regionY = board.bitboard->getRookRegion(bitlineY, x, 8); // 最后一个参数放8，获取一行就代8进去
    for (int x2 = x + 1; x2 < regionY[1]; x2++)
        result.emplace_back(Move{x, y, x2, y});
    if (board.teamOn(regionY[1], y) != team)
        result.emplace_back(Move{x, y, regionY[1], y});
    for (int x2 = x - 1; x2 > regionY[0]; x2--)
        result.emplace_back(Move{x, y, x2, y});
    if (board.teamOn(regionY[0], y) != team)
        result.emplace_back(Move{x, y, regionY[0], y});

    return result;
}

MOVES Moves::cannon(TEAM team, Board &board, int x, int y) // 炮的四个值分别对应eaten1, start, end, eaten2，若没有eaten则eaten = start或者end
{
    MOVES result{};
    result.reserve(64);

    // 横向着法
    BITLINE bitlineY = board.getBitLineY(y);
    REGION_CANNON regionY = board.bitboard->getCannonRegion(bitlineY, x, 8);
    for (int x2 = x + 1; x2 <= regionY[2]; x2++)
        result.emplace_back(Move{x, y, x2, y});
    if (board.teamOn(regionY[3], y) == -team && regionY[3] != regionY[2])
        result.emplace_back(Move{x, y, regionY[3], y});
    for (int x2 = x - 1; x2 >= regionY[1]; x2--)
        result.emplace_back(Move{x, y, x2, y});
    if (board.teamOn(regionY[0], y) == -team && regionY[0] != regionY[1])
        result.emplace_back(Move{x, y, regionY[0], y});

    // 纵向着法
    BITLINE bitlineX = board.getBitLineX(x);
    REGION_CANNON regionX = board.bitboard->getCannonRegion(bitlineX, y, 9);
    for (int y2 = y + 1; y2 <= regionX[2]; y2++)
        result.emplace_back(Move{x, y, x, y2});
    if (board.teamOn(x, regionX[3]) == -team && regionX[3] != regionX[2])
        result.emplace_back(Move{x, y, x, regionX[3]});
    for (int y2 = y - 1; y2 >= regionX[1]; y2--)
        result.emplace_back(Move{x, y, x, y2});
    if (board.teamOn(x, regionX[0]) == -team && regionX[0] != regionX[1])
        result.emplace_back(Move{x, y, x, regionX[0]});

    return result;
}
```


未完待续...
