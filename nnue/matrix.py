import numpy
import json

EMPTY = 0
KING = 1
GUARD = 2
BISHOP = 3
KNIGHT = 4
ROOK = 5
CANNON = 6
PAWN = 7
RED = 1
BLACK = 2

# NNUE文件
class NNUE_File:
    file = None
    jsonContent = None

    def __init__(self, filename):
        self.file = open(filename)

    def parse(self):
        self.jsonContent = self.jsonContent or json.loads(self.file.read())
        return self.jsonContent

# fen转numpy矩阵
def fen_to_matrix(fenCode: str):
    # 初始化2x7x90的零矩阵
    result = numpy.zeros((2, 7, 90), dtype=numpy.int8)

    PAIRS = {
        "K": (RED, KING), "k": (BLACK, KING),
        "A": (RED, GUARD), "a": (BLACK, GUARD),
        "B": (RED, BISHOP), "b": (BLACK, BISHOP),
        "N": (RED, KNIGHT), "n": (BLACK, KNIGHT),
        "R": (RED, ROOK), "r": (BLACK, ROOK),
        "C": (RED, CANNON), "c": (BLACK, CANNON),
        "P": (RED, PAWN), "p": (BLACK, PAWN)
    }

    x, y = 0, 0  # 棋盘坐标初始化
    for char in fenCode:
        if char == '/':
            # 换行
            x += 1
            y = 0
        elif char.isdigit():
            # 跳过空位
            y += int(char)
        else:
            # 处理棋子
            if char in PAIRS:
                color, piece = PAIRS[char]
                pos = position_2d_to_1d(x, y)
                result[color-1][piece-1][pos] = 1
            y += 1
    return result

# 棋盘二维坐标转一维坐标
def position_2d_to_1d(x, y):
    return 10 * x + y

# 棋盘一维坐标转二维坐标
def position_1d_to_2d(pos):
    return {"x": pos // 10, "y": pos % 10}

# 棋盘局势类, 一个2 * 7 * 90矩阵
class Situation:
    matrix = None
    vl = None

    # 初始化
    def __init__(self, fenCode: str, vl):
        self.matrix = fen_to_matrix(fenCode)
        self.vl = vl

    # 棋盘上下翻转
    def mirror_topbottom(self):
        new_matrix = self.matrix
        for k_team, v_team in self.matrix:
            for k_piece, v_piece in v_team:
                pass

    # 棋盘左右翻转
    def mirror_leftright(self):
        pass
