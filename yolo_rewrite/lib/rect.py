import numpy as np


def matrix_iou(a, b):
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i)


class Rect(tuple):
    __slots__ = list()

    def __new__(cls, rect):
        return super().__new__(cls, rect)

    def LTRB(left, top, right, bottom):
        return Rect((left, top, right, bottom))

    def LTWH(left, top, width, height):
        return Rect((
            left,
            top,
            left + width,
            top + height))

    def XYWH(center_x, center_y, width, height):
        return Rect((
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2))

    def astype(self, dtype):
        return Rect(map(dtype, self))

    @property
    def left(self):
        return self[0]

    @property
    def top(self):
        return self[1]

    @property
    def right(self):
        return self[2]

    @property
    def bottom(self):
        return self[3]

    @property
    def width(self):
        return self.right - self.left

    @property
    def height(self):
        return self.bottom - self.top

    def __and__(self, other):
        left = max(self.left, other.left)
        top = max(self.top, other.top)
        right = min(self.right, other.right)
        bottom = min(self.bottom, other.bottom)

        if left < right and top < bottom:
            return Rect.LTRB(left, top, right, bottom)
        else:
            return None

    def __mul__(self, k):
        try:
            kx, ky = k
        except ValueError:
            kx, ky = k, k
        return Rect.LTRB(
            self.left * kx,
            self.top * ky,
            self.right * kx,
            self.bottom * ky)

    @property
    def area(self):
        return self.width * self.height

    def iou(self, other):
        intersect = self & other
        if intersect is None:
            return 0
        return intersect.area \
            / (self.area + other.area - intersect.area)
