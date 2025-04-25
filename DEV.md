# Chess98

以下是关于代码细节的一些说明，注释里可能确实讲不清楚思路，在这里提供一下我设计的想法

## base.hpp

定义了代码所用到的常量和基本类，有以下这些：

- INF, BAN, ILLEGAL_VAL, U64
- PIECE_INDEX 整数类型，棋盘初始化时会生成棋子列表 Board.pieces，每一个 PIECE_INDEX 类型就相当于一个对特定 Piece 的引用，记录着这个 Piece 在 Board.pieces 里的索引号，若棋子不存在则为 EMPTY_INDEX
- PIECEID 整数类型，一个棋子的类别，比如黑车: -5，红帅: 1，不存在则为EMPTY_PIECEID，越界则为 OVERFLOW_PIECEID
- TEAM 整数类型，队伍类别，和 PIECEID 情况大致相同
- PIECEID_MAP 一个 9x10 格式的数组，代表中国象棋棋盘
- PIECE_NAME_PAIRS, NAME_PIECE_PAIRS

- Piece 类
    - team()
    - pieceid
    - x
    - y
    - pieceIndex
    - isLive: bool
- Move 类
    - id: int 快速比较着法
    - val
    - moveType
    - ==, !=
- Result 类
    - val
    - move

未完待续...
