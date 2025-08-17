# base.py
import numpy as np

# 棋子编码
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


def fen_to_matrix(fen: str) -> np.ndarray:
    """将中国象棋 FEN 字符串转换为 7×9×10 的输入张量"""
    PIECE_TO_INT = {
        'K': R_KING, 'A': R_GUARD, 'B': R_BISHOP, 'N': R_KNIGHT, 'R': R_ROOK, 'C': R_CANNON, 'P': R_PAWN,
        'k': B_KING, 'a': B_GUARD, 'b': B_BISHOP, 'n': B_KNIGHT, 'r': B_ROOK, 'c': B_CANNON, 'p': B_PAWN,
    }

    board = np.zeros((10, 9), dtype=int)  # 10行（y），9列（x）
    rows = fen.split('/')
    if len(rows) != 10:
        raise ValueError("FEN 棋盘部分应为10行，用'/'分隔")

    # 去除最后的空格和多余信息（如 ' w - - 0 1'）
    rows[-1] = rows[-1].strip().split()[0]

    for y in range(10):
        x = 0
        for c in rows[y]:
            if c.isdigit():
                x += int(c)
            elif c in PIECE_TO_INT:
                if x < 9:
                    board[y, x] = PIECE_TO_INT[c]
                    x += 1
                else:
                    raise ValueError(f"列溢出：y={y}, x={x}, 棋子={c}")
            else:
                raise ValueError(f"未知字符: {c}")

    # 构建 7 通道输出：每个通道对应一种棋子类型
    output = np.zeros((7, 9, 10), dtype=int)  # (通道, x, y)

    for y in range(10):
        for x in range(9):
            piece = board[y, x]
            if piece == EMPTY:
                continue
            abs_p = abs(piece)
            channel = abs_p - 1  # K=0, A=1, ..., P=6
            sign = 1 if piece > 0 else -1
            output[channel, x, y] = sign

    return output


class Situation:
    """棋局状态类，支持翻转增强"""
    def __init__(self, fen: str):
        self.fen = fen
        self.matrix = fen_to_matrix(fen)

    def __str__(self):
        return f"Situation(fen='{self.fen[:30]}...', matrix_shape={self.matrix.shape})"

    def flip_leftright(self):
        """左右翻转：x轴反转（in-place）"""
        self.matrix = self.matrix[:, ::-1, :]
        return self

    def flip_updown(self):
        """上下翻转：y轴反转（in-place）"""
        self.matrix = self.matrix[:, :, ::-1]
        return self