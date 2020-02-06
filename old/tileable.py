import os
import sys

currentDir = os.path.dirname(os.path.realpath(__file__))
if currentDir not in sys.path: sys.path.append(currentDir)
os.chdir(currentDir)

import numpy
from datetime import datetime
from PIL import Image

numpy.set_printoptions(linewidth=600, precision=3, edgeitems=10, threshold=4000,
                       suppress=True)


def getTime(start=None, header=''):
    current = datetime.now()
    if start is None: return current
    end = current - start
    header = '{}: '.format(header) if header != '' else ''
    print('{}{:>3}.{:3} ms'.format(header, *str(
        end.seconds * 1000 + end.microseconds / 1000).split('.')))
    return end.seconds * 1000 + end.microseconds / 1000


class World:
    def __init__(self, name: str, node, size=64, seed=1, **kwargs):
        self.name = name
        self.node = node
        
        self.size = size
        self.seed = seed
        self.kwargs = kwargs
        
        self.minX, self.maxX = 0, 0
        self.minY, self.maxY = 0, 0
        
        self.data = {}
        self.nodes = {}
    
    def create(self, r):
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                self.createNode(i, j)
    
    def createNode(self, x, y):
        self.minX = min(x, self.minX)
        self.minY = min(y, self.minY)
        self.maxX = max(x, self.maxX)
        self.maxY = max(y, self.maxY)
        
        self.nodes['{},{}'.format(x, y)] = self.node(self, x, y, **self.kwargs)
    
    def getNode(self, x, y):
        try:
            return self.nodes['{},{}'.format(x, y)]
        except KeyError:
            return None
    
    def getData(self, key):
        try:
            return self.data[key]
        except KeyError:
            return None
    
    def getNeighbors(self, x, y):
        return [self.getNode(x + i, y + j) if i != 0 or j != 0 else None for j
                in [-1, 0, 1] for i in [-1, 0, 1]]
    
    def toImage(self, sections=None, export=False):
        box = ((self.maxX - self.minX + 1) * self.size,
               (self.maxY - self.minY + 1) * self.size)
        stitched = Image.new('RGB', box)
        for key, node in self.nodes.items():
            x, y = map(int, key.split(','))
            box = ((x - self.minX) * self.size, (y - self.minY) * self.size)
            stitched.paste(im=node.toImage(sections, export), box=box)
        stitched.save(self.name + '.png', 'PNG')


class Node:
    def __init__(self, world, x, y, **kwargs):
        self.world = world
        
        self.x = x
        self.y = y
        
        self.neighbors = world.getNeighbors(x, y)
        
        self.setSeed(x, y)
        
        self.array = self.createArray()
        
        t = getTime()
        freq = 1
        amp = 1
        maxVal = 0
        for _ in range(kwargs.get('octaves', 1)):
            self.array += self.generate(freq, amp)
            freq *= 2
            maxVal += amp
            amp *= kwargs.get('persistence', 0.5)
        self.array /= maxVal
        getTime(t, 'Gen Time for ({:2},{:2})'.format(x, y))
        
        numpy.clip(self.array, 0, 1, out=self.array)
    
    def getSeed(self):
        return self.world.seed
    
    def setSeed(self, x, y):
        h = 0
        for c in '{},{}:{}'.format(x, y, self.getSeed()):
            h = (31 * h + 17 * ord(c)) & 0xFFFFFFFF
        numpy.random.seed(h)
    
    def createArray(self):
        return numpy.zeros(shape=(world.size, world.size))
    
    def generate(self, freq, amp):
        return numpy.zeros(shape=(self.world.size, self.world.size))
    
    def toImage(self, sections=None, export=False):
        array = 255 * self.array
        if sections is not None:
            sections = 256 / sections
            array = array // sections * sections
        img = Image.fromarray(array.astype('uint8'))
        if export:
            img.save('{}.({},{}).png'.format(self.world.name, self.x, self.y),
                     'PNG')
        return img


class DSNode(Node):
    def __init__(self, world, x, y, period=2 ** 5, func=None, **kwargs):
        if world.getData('func') is None:
            # noinspection PyUnusedLocal
            def defaultFunc(array, size, x, y, row, col, r, h, p, rand):
                return min(max(h + (rand - 0.5) * r / p, 0.0), 1.0)
            
            world.data['func'] = defaultFunc if func is None else func
        
        self.period = period
        self.func = world.getData('func')
        
        self.corners = [None] * 4
        
        super().__init__(world, x, y, **kwargs)
        
        self.rawArray = numpy.copy(self.array)
        self.array = self.array[:size, :size]
    
    def getSeed(self):
        return self.world.seed * self.period
    
    def createArray(self):
        r = 2
        while True:
            if self.world.size <= r: break
            r *= 2
        return numpy.zeros(shape=(r + 1, r + 1))
    
    def generate(self, freq, amp):
        array = self.createArray() - 1
        newSize = array.shape[0]
        r = newSize - 1
        
        if self.neighbors[1] is not None:
            array[0, :] = self.neighbors[1].rawArray[-1, :]
        if self.neighbors[3] is not None:
            array[:, 0] = self.neighbors[3].rawArray[:, -1]
        if self.neighbors[5] is not None:
            array[:, -1] = self.neighbors[5].rawArray[0, :]
        if self.neighbors[7] is not None:
            array[-1, :] = self.neighbors[7].rawArray[0, :]
        
        if array[0][0] == -1: array[0][0] = self.getCornerValue([0, 1, 3],
                                                                [3, 2, 1])
        if array[0][r] == -1: array[0][r] = self.getCornerValue([1, 2, 5],
                                                                [3, 2, 0])
        if array[r][0] == -1: array[r][0] = self.getCornerValue([3, 6, 7],
                                                                [3, 1, 0])
        if array[r][r] == -1: array[r][r] = self.getCornerValue([5, 7, 8],
                                                                [2, 1, 0])
        
        self.corners[0] = array[0][0]
        self.corners[1] = array[0][r]
        self.corners[2] = array[r][0]
        self.corners[3] = array[r][r]
        
        r //= 2
        while r > 0:
            for row in range(r, newSize, 2 * r):
                for col in range(r, newSize, 2 * r):
                    self.diamond(array, row, col, r)
            switch = False
            for row in range(0, newSize, r):
                for col in range(0, newSize, r):
                    if switch:
                        self.square(array, row, col, r)
                    switch = not switch
            r //= 2
        
        return array
    
    def getCornerValue(self, neighbors, corners):
        for i, neighbor in enumerate(neighbors):
            if self.neighbors[neighbor] is not None:
                if self.neighbors[neighbor].corners[corners[i]] is not None:
                    return self.neighbors[neighbor].corners[corners[i]]
        return numpy.random.random()
    
    def setValue(self, array, row, col, r, h):
        if array[row][col] == -1:
            array[row][col] = self.func(array, self.world.size, self.x, self.y,
                                        row, col, r, h, self.period,
                                        numpy.random.random())
    
    def diamond(self, array, row, col, r):
        vals = [
            array[row - r][col - r],
            array[row - r][col + r],
            array[row + r][col - r],
            array[row + r][col + r]
        ]
        self.setValue(array, row, col, r, sum(vals) / len(vals))
    
    def square(self, array, row, col, r):
        vals = []
        if 0 <= row - r:
            vals.append(array[row - r][col])
        elif self.neighbors[1] is not None:
            vals.append(self.neighbors[1].array[row - r][col])
        if 0 <= col - r:
            vals.append(array[row][col - r])
        elif self.neighbors[3] is not None:
            vals.append(self.neighbors[3].array[row][col - r])
        if row + r < array.shape[0]:
            vals.append(array[row + r][col])
        elif self.neighbors[7] is not None:
            vals.append(self.neighbors[7].array[row + r - self.world.size][col])
        if col + r < array.shape[1]:
            vals.append(array[row][col + r])
        elif self.neighbors[5] is not None:
            vals.append(self.neighbors[5].array[row][col + r - self.world.size])
        self.setValue(array, row, col, r, sum(vals) / len(vals))


class PerlinNode(Node):
    def __init__(self, world, x, y, step=45, **kwargs):
        if world.getData('grad') is None:
            world.data['grad'] = numpy.array(
                [[numpy.cos(numpy.radians(ang)), numpy.sin(numpy.radians(ang))]
                 for ang in range(0, 360, step)])
        
        self.step = step
        self.grad = world.getData('grad')
        
        super().__init__(world, x, y, **kwargs)
    
    def getSeed(self):
        return self.world.seed * self.step
    
    def generate(self, freq, amp):
        lin = numpy.linspace(0, freq, self.world.size, endpoint=False)
        x, y = numpy.meshgrid(lin, lin)
        
        p = numpy.zeros(shape=(freq + 1, freq + 1), dtype=int)
        for row in range(0, freq + 1):
            for col in range(0, freq + 1):
                self.setSeed(self.x * freq + col, self.y * freq + row)
                p[row][col] = numpy.random.randint(0, 360 // self.step)
        
        xi = x.astype(int)
        yi = y.astype(int)
        
        xf = x - xi
        yf = y - yi
        
        u = self.fade(xf)
        v = self.fade(yf)
        
        n0 = self.gradient(p[yi, xi], xf, yf)
        n1 = self.gradient(p[yi, xi + 1], xf - 1, yf)
        x1 = self.lerp(n0, n1, u)
        
        n0 = self.gradient(p[yi + 1, xi], xf, yf - 1)
        n1 = self.gradient(p[yi + 1, xi + 1], xf - 1, yf - 1)
        x2 = self.lerp(n0, n1, u)
        
        return (self.lerp(x1, x2, v) + 0.5) * amp
    
    def gradient(self, h, x, y):
        """grad converts h to the right gradient vector and return the dot product with (x,y)"""
        g = self.grad[h]
        return g[:, :, 0] * x + g[:, :, 1] * y
    
    @staticmethod
    def lerp(a, b, x):
        """linear interpolation"""
        return a + x * (b - a)
    
    @staticmethod
    def fade(t):
        """6t^5 - 15t^4 + 10t^3"""
        return t * t * t * (t * (t * 6 - 15) + 10)


class WorleyNode(Node):
    def __init__(self, world, x, y, count=16, correction=0.75, **kwargs):
        self.count = count
        self.correction = correction
        
        super().__init__(world, x, y, **kwargs)
    
    def getSeed(self):
        return self.world.seed * self.count
    
    def generate(self, freq, amp):
        points = numpy.empty((0, 2))
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                self.setSeed(self.x + x, self.y + y)
                p = numpy.random.random(size=(self.count * freq, 2))
                points = numpy.vstack((points, numpy.add(p, [x, y])))
        
        array = numpy.ones(shape=(self.world.size, self.world.size))
        
        lin = numpy.linspace(0, 1, self.world.size, endpoint=False)
        x, y = numpy.meshgrid(lin, lin)
        
        for _, value in enumerate(
                numpy.sqrt((x - p[0]) * (x - p[0]) + (y - p[1]) * (y - p[1]))
                for p in points):
            array = numpy.minimum(value, array)
        
        return array * numpy.sqrt(self.count * freq) * self.correction * amp


if __name__ == '__main__':
    size = 128
    seed = 2
    octaves = 2
    persistence = 0.5
    
    period = 2 ** 3
    
    step = 30
    
    count = 16
    correction = 0.75
    
    
    # noinspection PyUnusedLocal
    def func(array, size, x, y, row, col, r, h, p, rand):
        xVal = x * size + col - size / 2
        yVal = y * size + row - size / 2
        dist = numpy.sqrt(
            xVal * xVal + yVal * yVal) if xVal != 0 or yVal != 0 else 0.0000001
        val = h + (rand - 0.5) * r / p * size / dist
        return min(max(val, 0.0), 1.0)
    
    
    # world = World('DSNode', size = size, node = DSNode, seed = 1, period = period, func = func)
    # world.create(5)
    # world.toImage()
    
    world = World('PerlinNode', size=size, node=PerlinNode, seed=seed,
                  octaves=octaves, persistence=persistence, step=step)
    world.create(2)
    world.toImage()
    
    # world = World('WorleyNode', size = size, node = WorleyNode, seed = seed, octaves = octaves, persistence = persistence, count = count, correction = correction)
    # world.create(2)
    # world.toImage()
