"""
매개변수를 갱신할 클래스들 모음
"""
import numpy as np


class SDG:
    def __init__(self, lr = 0.01):
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
