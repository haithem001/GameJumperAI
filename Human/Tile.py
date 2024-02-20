class tile:

    def __init__(self, x, y, w, h):
        self.h = h
        self.w = w
        self.y = y
        self.x = x

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getH(self):
        return self.h

    def getW(self):
        return self.w

    def setX(self, x):
        self.x = x

    def setY(self, y):
        self.y = y