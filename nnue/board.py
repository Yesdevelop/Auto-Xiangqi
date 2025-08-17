# board.py
import numpy as np

# 棋子编码（保持不变）
EMPTY = 0
R_KING, R_GUARD, R_BISHOP, R_KNIGHT, R_ROOK, R_CANNON, R_PAWN = 1, 2, 3, 4, 5, 6, 7
B_KING, B_GUARD, B_BISHOP, B_KNIGHT, B_ROOK, B_CANNON, B_PAWN = -1, -2, -3, -4, -5, -6, -7
Red, Black = 1, 0

def fen_to_matrix(fen: str) -> np.ndarray:
    """将中国象棋 FEN 字符串转换为 7×9×10 的输入张量"""
    PIECE_TO_INT = {
        'K': R_KING, 'A': R_GUARD, 'B': R_BISHOP, 'N': R_KNIGHT, 'R': R_ROOK, 'C': R_CANNON, 'P': R_PAWN,
        'k': B_KING, 'a': B_GUARD, 'b': B_BISHOP, 'n': B_KNIGHT, 'r': B_ROOK, 'c': B_CANNON, 'p': B_PAWN,
    }

    board = np.zeros((10, 9), dtype=int)
    rows = fen.split('/')
    if len(rows) != 10:
        raise ValueError("FEN 棋盘部分应为10行，用'/'分隔")

    rows[-1] = rows[-1].strip().split()[0]  # 去除元信息

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

    output = np.zeros((7, 9, 10), dtype=int)
    for y in range(10):
        for x in range(9):
            piece = board[y, x]
            if piece == EMPTY:
                continue
            channel = abs(piece) - 1
            sign = 1 if piece > 0 else -1
            output[channel, x, y] = sign

    return output

class Situation:
    """棋局状态类，支持翻转增强 → 返回新对象"""
    def __init__(self, fen: str):
        self.fen = fen
        self.actor_flag = Red if "w" in fen else Black
        self.matrix = fen_to_matrix(fen)

    def __str__(self):
        return f"Situation(actor_flag={self.actor_flag}, matrix_shape={self.matrix.shape})"

    def flip_left_and_right(self):
        """左右翻转：返回新对象"""
        new = object.__new__(Situation)
        new.fen = self.fen
        new.actor_flag = self.actor_flag
        new.matrix = self.matrix[:, ::-1, :].copy()  # x轴反转
        return new

    def flip_up_and_down(self):
        """上下翻转 + 颜色反转 + 切换走子方：返回新对象"""
        new = object.__new__(Situation)
        new.fen = self.fen
        new.actor_flag = 1 - self.actor_flag  # 轮到对方走
        new.matrix = (-self.matrix)[:, :, ::-1].copy()  # 颜色反转 + y轴反转
        return new

if __name__ == "__main__":
    test_fen = 'rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0'
    s1 = Situation(fen=test_fen)
    print(s1.matrix[0],s1.actor_flag)
    print()
    s2 = s1.flip_left_and_right()
    print(s2.matrix[0],s2.actor_flag)
    print()
    s3 = s1.flip_up_and_down()
    print(s3.matrix[0],s3.actor_flag)
    print()
