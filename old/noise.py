import os
import sys

currentDir = os.path.dirname(os.path.realpath(__file__))
if currentDir not in sys.path: sys.path.append(currentDir)
os.chdir(currentDir)

import numpy
import datetime
from PIL import Image

numpy.set_printoptions(linewidth=600, precision=3, edgeitems=10, threshold=4000,
                       suppress=True)


def getTime(start=None, header=''):
    current = datetime.datetime.now()
    if start is None: return current
    end = current - start
    header = '{}: '.format(header) if header != '' else ''
    print('{}{:>3}.{:3} ms'.format(header, *str(
        end.seconds * 1000 + end.microseconds / 1000).split('.')))
    return end.seconds * 1000 + end.microseconds / 1000


class Noise:
    def __init__(self, name, size=64, seed=1):
        self.name = name
        self.size = size
        self.seed = seed
        
        self.setSeed()
        
        self.array = None
        
        self.generate()
        
        numpy.clip(self.array, 0, 1, out=self.array)
    
    def getSeed(self):
        return self.seed
    
    def setSeed(self):
        numpy.random.seed(self.getSeed() & 0xFFFFFFFF)
    
    def generate(self):
        self.array = numpy.zeros(shape=(self.size, self.size))
    
    def toImage(self, sections=None, export=False):
        array = 255 * self.array
        if sections is not None:
            sections = 256 / sections
            array = array // sections * sections
        img = Image.fromarray(array.astype('uint8'))
        if export:
            img.save(self.name + '.png', 'PNG')
        return img


class DSNoise(Noise):
    def __init__(self, name, size=64, period=2 ** 5, seed=1, func=None):
        def defaultFunc(array, x, y, r, h, p, rand):
            return min(max(h + (rand - 0.5) * r / p, 0.0), 1.0)
        
        self.newSize = size
        self.period = period
        self.func = defaultFunc if func is None else func
        
        super().__init__(name, size, seed)
        
        self.array = self.array[:size, :size]
    
    def getSeed(self):
        return self.seed * self.period
    
    def generate(self):
        r = 2
        while True:
            if self.size <= r: break
            r *= 2
        
        self.newSize = r + 1
        self.array = numpy.zeros(shape=(self.newSize, self.newSize))
        
        self.array[0][0] = numpy.random.random()
        self.array[0][r] = numpy.random.random()
        self.array[r][0] = numpy.random.random()
        self.array[r][r] = numpy.random.random()
        
        r //= 2
        while r > 0:
            for x in range(r, self.newSize, 2 * r):
                for y in range(r, self.newSize, 2 * r):
                    self.diamond(x, y, r)
            switch = False
            for x in range(0, self.newSize, r):
                for y in range(0, self.newSize, r):
                    if switch:
                        self.square(x, y, r)
                    switch = not switch
            r //= 2
    
    def setValue(self, x, y, r, h):
        self.array[x][y] = self.func(self.array, x, y, r, h, self.period,
                                     numpy.random.random())
    
    def diamond(self, x, y, r):
        vals = [
            self.array[x - r][y - r],
            self.array[x - r][y + r],
            self.array[x + r][y - r],
            self.array[x + r][y + r]
        ]
        self.setValue(x, y, r, sum(vals) / len(vals))
    
    def square(self, x, y, r):
        vals = []
        if 0 <= x - r:
            vals.append(self.array[x - r][y])
        if 0 <= y - r:
            vals.append(self.array[x][y - r])
        if x + r < self.array.shape[0]:
            vals.append(self.array[x + r][y])
        if y + r < self.array.shape[1]:
            vals.append(self.array[x][y + r])
        self.setValue(x, y, r, sum(vals) / len(vals))


class PerlinNoise(Noise):
    def __init__(self, name, size=64, rnge=(0, 10), seed=1, step=45):
        self.rnge = rnge
        self.step = step
        self.grad = numpy.array(
            [[numpy.cos(numpy.radians(ang)), numpy.sin(numpy.radians(ang))] for
             ang in range(0, 360, step)])
        
        super().__init__(name, size, seed)
    
    def generate(self):
        rnge = self.rnge[1] - self.rnge[0] + 1
        
        p = numpy.zeros(shape=(rnge, rnge), dtype=int)
        for y in range(self.rnge[0], self.rnge[1] + 1):
            for x in range(self.rnge[0], self.rnge[1] + 1):
                h = 0
                for ch in '{},{}'.format(x, y):
                    h = (31 * h + ord(ch)) & 0xFFFFFFFF
                h = ((h + 0x80000000) & 0xFFFFFFFF) - 0x80000000
                
                numpy.random.seed((self.seed * self.step * h) & 0xFFFFFFFF)
                p[x][y] = numpy.random.randint(0, 360)
        
        lin = numpy.linspace(*self.rnge, self.size, endpoint=False)
        x, y = numpy.meshgrid(lin, lin)
        
        xi = x.astype(int)
        yi = y.astype(int)
        xf = x - xi
        yf = y - yi
        
        n00 = self.gradient(p[xi, yi], xf, yf)
        n01 = self.gradient(p[xi, yi + 1], xf, yf - 1)
        n11 = self.gradient(p[xi + 1, yi + 1], xf - 1, yf - 1)
        n10 = self.gradient(p[xi + 1, yi], xf - 1, yf)
        
        # fade factors
        u = self.fade(xf)
        v = self.fade(yf)
        # combine noises
        x1 = self.lerp(n00, n10, u)
        x2 = self.lerp(n01, n11, u)  # FIX1: I was using n10 instead of n01
        self.array = self.lerp(x1, x2,
                               v) + 0.5  # FIX2: I also had to reverse x1 and x2 here
    
    @staticmethod
    def lerp(a, b, x):
        """linear interpolation"""
        return a + x * (b - a)
    
    @staticmethod
    def fade(t):
        """6t^5 - 15t^4 + 10t^3"""
        # return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def gradient(self, h, x, y):
        """grad converts h to the right gradient vector and return the dot product with (x,y)"""
        g = self.grad[h % (360 // self.step)]
        return g[:, :, 0] * x + g[:, :, 1] * y


class WorleyNoise(Noise):
    def __init__(self, name, size=64, count=16, seed=1, correction=0.75):
        self.count = count
        self.correction = correction
        
        super().__init__(name, size, seed)
    
    def getSeed(self):
        return self.seed * self.count
    
    def generate(self):
        points = numpy.random.random(size=(self.count, 2))
        
        self.array = numpy.ones(shape=(size, size))
        
        lin = numpy.linspace(0, 1, self.size)
        x, y = numpy.meshgrid(lin, lin)
        
        for i, value in enumerate(
                numpy.sqrt((x - p[0]) * (x - p[0]) + (y - p[1]) * (y - p[1]))
                for p in points):
            self.array = numpy.minimum(value, self.array)
        
        self.array *= numpy.sqrt(self.count) * self.correction


if __name__ == '__main__':
    size = 128
    seed = 2
    octaves = 8
    persistence = 0.5
    
    period = 2 ** 3
    
    step = 30
    
    count = 16
    correction = 0.75
    
    # noise = DSNoise('DSNoise', size = size, period = period, seed = seed)
    # noise.toImage(export = True)
    
    # noise = PerlinNoise('PerlinNoise', size = size, rnge = rnge, seed = seed, step = step)
    # noise.toImage(export = True)
    
    # noise = WorleyNoise('WorleyNoise', size = size, seed = seed, count = count, correction = correction)
    # noise.toImage(export = True)
