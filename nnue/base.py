import numpy as np

EMPTY = 0

R_KING = 1
R_GUARD = 2
R_BISHOP = 3
R_KNIGHT = 4
R_ROOK = 5
R_CANNON = 6
R_PAWN = 7

B_KING = -1
B_GUARD = -2
B_BISHOP = -3
B_KNIGHT = -4
B_ROOK = -5
B_CANNON = -6
B_PAWN = -7

# fen转矩阵，7 * 9 * 10
def fen_to_matrix(fen: str) -> np.ndarray:
    # 棋盘：10行（y: 0~9，从下到上），9列（x: 0~8，从左到右）
    PIECE_TO_INT = {
        'K': R_KING,     # 红帅
        'A': R_GUARD,    # 红仕
        'B': R_BISHOP,   # 红相
        'N': R_KNIGHT,   # 红马
        'R': R_ROOK,     # 红车
        'C': R_CANNON,   # 红炮
        'P': R_PAWN,     # 红兵

        'k': B_KING,     # 黑将
        'a': B_GUARD,    # 黑仕
        'b': B_BISHOP,   # 黑相
        'n': B_KNIGHT,   # 黑马
        'r': B_ROOK,     # 黑车
        'c': B_CANNON,   # 黑炮
        'p': B_PAWN,     # 黑兵
    }

    board = np.zeros((10, 9), dtype=int)

    rows = fen.split('/')
    if len(rows) != 10:
        raise ValueError("FEN 棋盘部分应该由 10 行组成，用 '/' 分隔")

    for y in range(10):
        row_str = rows[y]
        x = 0
        for c in row_str:
            if c.isdigit():
                num_empty = int(c)
                x += num_empty
            else:
                piece_int = PIECE_TO_INT.get(c, EMPTY)
                if x < 9:
                    board[y, x] = piece_int
                    x += 1
                else:
                    raise ValueError(f"棋盘列溢出：y={y}, x={x}, 棋子={c}")

    # 初始化输出矩阵：7个通道（红方棋子类型），每个是 9x10
    output = np.zeros((7, 9, 10), dtype=int)  # shape: (7, 9, 10)

    for y in range(10):     # 棋盘行 0~9 （从下到上）
        for x in range(9):  # 棋盘列 0~8 （从左到右）
            piece = board[y, x]
            if piece == EMPTY:
                continue
            # 判断是哪种棋子，并设置对应通道
            if piece == R_KING:
                output[0, x, y] = 1
            elif piece == R_GUARD:
                output[1, x, y] = 1
            elif piece == R_BISHOP:
                output[2, x, y] = 1
            elif piece == R_KNIGHT:
                output[3, x, y] = 1
            elif piece == R_ROOK:
                output[4, x, y] = 1
            elif piece == R_CANNON:
                output[5, x, y] = 1
            elif piece == R_PAWN:
                output[6, x, y] = 1
            elif piece == B_KING:
                output[0, x, y] = -1
            elif piece == B_GUARD:
                output[1, x, y] = -1
            elif piece == B_BISHOP:
                output[2, x, y] = -1
            elif piece == B_KNIGHT:
                output[3, x, y] = -1
            elif piece == B_ROOK:
                output[4, x, y] = -1
            elif piece == B_CANNON:
                output[5, x, y] = -1
            elif piece == B_PAWN:
                output[6, x, y] = -1

    return output

# 局势类
class Situation:
    fen = None
    matrix = None

    # 初始化，初始matrix和fen
    def __init__(self, fen: str):
        self.fen = fen
        self.matrix = fen_to_matrix(fen)

    # 转字符串
    def __str__(self):
        return f"Situation(fen='{self.fen[:30]}...', matrix_shape={self.matrix.shape})"

    # 上下旋转
    def flip_updown(self):
        # 上下翻转棋盘 -> y轴反转
        self.matrix = self.matrix[:, :, ::-1]  # 只翻转 y轴 (axis=2)
        # 也要更新 fen（可选，这里暂不实现 fen 同步更新）

    # 左右旋转
    def flip_leftright(self):
        # 左右翻转棋盘 -> x轴反转
        self.matrix = self.matrix[:, ::-1, :]  # 只翻转 x轴 (axis=1)
