import datetime

import numpy as np
from PIL import Image

__author__ = 'Ryan Smith <smithrj5@vcu.edu>'
__version__ = '1.0.0'


def getTime(start=None, header=''):
    current = datetime.datetime.now()
    if start is None: return current
    end = current - start
    header = '{}: '.format(header) if header != '' else ''
    time = end.seconds * 1000000 + end.microseconds
    print('{}{:>6} us'.format(header, time))
    return time


class World:
    def __init__(self, name, tile, **kwargs):
        self.name = name
        self.tile = tile
        
        self.size = kwargs.get('size', 64)
        self.seed = kwargs.get('seed', 1)
        self.func = kwargs.get('worldFunc', lambda x, y, arr: arr)
        self.kwargs = kwargs
        
        self.minX, self.maxX = 0, 0
        self.minY, self.maxY = 0, 0
        
        self.data = {}
        self.tiles = {}
    
    def create(self, r):
        print('Generating World \'{}\' with radius {}'.format(self.name, r))
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                self.createTile(i, j)
    
    def createTile(self, x, y):
        self.minX = min(x, self.minX)
        self.minY = min(y, self.minY)
        self.maxX = max(x, self.maxX)
        self.maxY = max(y, self.maxY)
        
        self.tiles['{},{}'.format(x, y)] = self.tile(self, x, y, **self.kwargs)
    
    def getTile(self, x, y):
        try:
            return self.tiles['{},{}'.format(x, y)]
        except KeyError:
            return None
    
    def getData(self, key):
        try:
            return self.data[key]
        except KeyError:
            return None
    
    def getNeighbors(self, x, y):
        return [self.getTile(x + i, y + j) for j in [-1, 0, 1] for i in
                [-1, 0, 1] if i != 0 or j != 0]
    
    def toNormal(self, sections=256, export=False):
        # box = ((self.maxX - self.minX + 1) * self.size, (self.maxY - self.minY + 1) * self.size)
        # stitched = Image.new('RGB', box)
        # for key, tile in self.tiles.items():
        # x, y = map(int, key.split(','))
        # box = ((x - self.minX) * self.size, (y - self.minY) * self.size)
        # stitched.paste(im = tile.toNormal(export), box = box)
        # stitched.save('{} - {}.png'.format(self.name, 'normals'), 'PNG')
        pass
    
    def toImage(self, sections=256, export=False):
        box = ((self.maxX - self.minX + 1) * self.size,
               (self.maxY - self.minY + 1) * self.size)
        stitched = Image.new('RGB', box)
        for key, tile in self.tiles.items():
            x, y = map(int, key.split(','))
            box = ((x - self.minX) * self.size, (y - self.minY) * self.size)
            stitched.paste(im=tile.toImage(sections=sections, export=export),
                           box=box)
        stitched.save(self.name + '.png', 'PNG')


class Tile:
    def __init__(self, world, x, y, **kwargs):
        self.world = world
        
        self.x = x
        self.y = y
        
        self.neighbors = world.getNeighbors(x, y)
        
        self.setSeed(x, y)
        
        self.array = self.createArray()
        
        t = getTime()
        maxValue = 0
        amp = 1
        freq = 1
        for _ in range(kwargs.get('octaves', 1)):
            self.array += self.generate(freq, amp) * amp
            maxValue += amp
            amp *= kwargs.get('persistence', 0.5)
            freq *= 2
        self.array = self.world.func(x, y, self.array / maxValue)
        getTime(t, 'Gen Time for ({:2},{:2})'.format(x, y))
        
        np.clip(self.array, 0, 1, out=self.array)
        
        self.world = None
        self.neighbors = None
    
    def getSeed(self):
        return self.world.seed
    
    def setSeed(self, x, y):
        h = 0
        for c in '{},{}:{}'.format(x, y, self.getSeed()):
            h = (31 * h + 17 * ord(c)) & 0xFFFFFFFF
        np.random.seed(h)
    
    def createArray(self):
        return np.zeros(shape=(self.world.size, self.world.size))
    
    def generate(self, freq, amp):
        return np.zeros(shape=(self.world.size, self.world.size))
    
    def toImage(self, tint=(255, 255, 255), sections=256, export=False):
        # array = self.colorize(tint(self.array, tint), sections)
        array = self.colorize([self.array], sections)
        img = Image.fromarray(array, mode='RGBA')
        if export:
            img.save('{}.({},{}).png'.format(self.world.name, self.x, self.y),
                     'PNG')
        return img
    
    @staticmethod
    def tint(array, color=[1., 1., 1.]):
        return [array * color[i] for i in range(3)]
    
    @staticmethod
    def colorize(color, sections=256):
        # TODO - Add tint logic to this method
        if len(color) == 1:
            color = [color[0]] * 3
        sections = 256 / sections
        for i, c in enumerate(color):
            if type(c) in [int, np.int8, np.int16, np.int32, np.int64]:
                color[i] = np.uint32(c)
            elif type(c) in [float, np.float16, np.float32, np.float64]:
                color[i] = np.uint32(c * 255)
            elif type(c) == np.ndarray:
                if c.dtype in [np.int8, np.int16, np.int32, np.int64, np.uint8,
                               np.uint16, np.uint32, np.uint64]:
                    color[i] = c.astype('uint32')
                elif c.dtype in [np.float16, np.float32, np.float64]:
                    color[i] = (c * 255).astype('uint32')
            color[i] = ((color[i] & 0xFF) // sections * sections).astype(
                'uint32')
        while len(color) < 4:
            color.append(np.uint32(255))
        return (color[0] << 0) + (color[1] << 8) + (color[2] << 16) + (
                    color[3] << 24)
    
    @staticmethod
    def colorizeTint(array, color=[1., 1., 1.], sections=256):
        # TODO - Add tint logic to this method
        if len(color) == 1:
            color = [color[0]] * 3
        sections = 256 / sections
        for i, c in enumerate(color):
            if type(c) in [int, np.int8, np.int16, np.int32, np.int64]:
                color[i] = np.uint32(c)
            elif type(c) in [float, np.float16, np.float32, np.float64]:
                color[i] = np.uint32(c * 255)
            elif type(c) == np.ndarray:
                if c.dtype in [np.int8, np.int16, np.int32, np.int64, np.uint8,
                               np.uint16, np.uint32, np.uint64]:
                    color[i] = c.astype('uint32')
                elif c.dtype in [np.float16, np.float32, np.float64]:
                    color[i] = (c * 255).astype('uint32')
            color[i] = ((color[i] & 0xFF) // sections * sections).astype(
                'uint32')
        while len(color) < 4:
            color.append(np.uint32(255))
        return (array * (
                    (color[0] << 0) + (color[1] << 8) + (color[2] << 16) + (
                        color[3] << 24))).astype('uint32')


class TileExperimental(Tile):
    def getNeighborPixels(self, x, y):
        list = []
        for i, j in [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1),
                     (-1, 1), (-1, 0)]:
            xPos = x + i
            yPos = y + j
            tile = self
            
            flagMinX = xPos < 0
            flagMaxX = self.world.size <= xPos
            flagMinY = yPos < 0
            flagMaxY = self.world.size <= yPos
            if flagMinX:
                if flagMinY:
                    tile = self.neighbors[0]
                elif flagMaxY:
                    tile = self.neighbors[5]
                    yPos -= self.world.size
                else:
                    tile = self.neighbors[3]
            elif flagMaxX:
                if flagMinY:
                    tile = self.neighbors[2]
                    xPos -= self.world.size
                elif flagMaxY:
                    tile = self.neighbors[7]
                    xPos -= self.world.size
                    yPos -= self.world.size
                else:
                    tile = self.neighbors[4]
                    xPos -= self.world.size
            else:
                if flagMinY:
                    tile = self.neighbors[1]
                elif flagMaxY:
                    tile = self.neighbors[6]
                    yPos -= self.world.size
                else:
                    tile = self
            
            if tile is None:
                tile = self
                xPos = x
                yPos = y
            
            list.append(np.array([i / self.world.size, j / self.world.size,
                                  tile.array[yPos][xPos] - self.array[y][x]]))
        return list
    
    def getNeighborVectors(self, x, y):
        neighbors = []
        for i, j in [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1),
                     (-1, 1), (-1, 0)]:
            xPos = x + i
            yPos = y + j
            array = self.array.copy()
            
            minX = xPos < 0
            maxX = self.world.size <= xPos
            minY = yPos < 0
            maxY = self.world.size <= yPos
            
            conditions = (((minX * minY) * 1) + (
                        (np.logical_not(minX + maxX) * minY) * 2) + (
                                      (maxX * minY) * 3) +
                          ((minX * np.logical_not(minY + maxY)) * 4) + ((
                                                                                    maxX * np.logical_not(
                                                                                minY + maxY)) * 5) +
                          ((minX * maxY) * 6) + ((np.logical_not(
                        minX + maxX) * maxY) * 7) + ((maxX * maxY) * 8))
            arrays = [
                self.array,
                (np.flip(self.neighbors[0].array) if self.neighbors[
                    0] else self.array),
                (np.flipud(self.neighbors[1].array) if self.neighbors[
                    1] else self.array),
                (np.flip(self.neighbors[2].array) if self.neighbors[
                    2] else self.array),
                (np.fliplr(self.neighbors[3].array) if self.neighbors[
                    3] else self.array),
                (np.fliplr(self.neighbors[4].array) if self.neighbors[
                    4] else self.array),
                (np.flip(self.neighbors[5].array) if self.neighbors[
                    5] else self.array),
                (np.flipud(self.neighbors[6].array) if self.neighbors[
                    6] else self.array),
                (np.flip(self.neighbors[7].array) if self.neighbors[
                    7] else self.array)
            ]
            
            array = np.choose(conditions, arrays)
            
            np.add(xPos, 1, out=xPos, where=minX)
            np.add(yPos, 1, out=yPos, where=minY)
            np.add(xPos, -1, out=xPos, where=maxX)
            np.add(yPos, -1, out=yPos, where=maxY)
            
            xDir = (x.copy() * 0 + i) / self.world.size
            yDir = (y.copy() * 0 + j) / self.world.size
            
            neighbors.append(
                np.array([xDir, yDir, array[yPos, xPos] - self.array]))
        
        return np.moveaxis(np.array(neighbors), 1, 3)
    
    def toNormal(self, export=False):
        r = self.array.copy() * 0
        g = self.array.copy() * 0
        b = self.array.copy() * 0
        r1 = self.array.copy() * 0
        g1 = self.array.copy() * 0
        b1 = self.array.copy() * 0
        
        lin = np.arange(self.world.size, dtype=int)
        x, y = np.meshgrid(lin, lin)
        
        neighbors = self.getNeighborVectors(x, y)
        # print(neighbors)
        for i, p2 in enumerate(neighbors):
            p1 = neighbors[i - 1]
            
            cross = np.cross(p1[y, x], p2[y, x])
            cross = (cross / np.linalg.norm(cross, axis=2).reshape(
                (self.world.size, self.world.size, 1)) + 1) / 2
            
            r += cross[:, :, 0]
            g += cross[:, :, 1]
            b += (cross[:, :, 2] + 1) / 2
        r, g, b = r / 8, g / 8, b / 8
        
        for y in range(self.world.size):
            for x in range(self.world.size):
                neighbors = self.getNeighborPixels(x, y)
                # print(neighbors)
                for i, p2 in enumerate(neighbors):
                    p1 = neighbors[i - 1]
                    p0 = self.array[y][x]
                    
                    p2[2] -= p0
                    p1[2] -= p0
                    
                    cross = np.cross(p1, p2)
                    cross = (cross / np.linalg.norm(cross) + 1) / 2
                    
                    r1[y][x] += cross[0]
                    g1[y][x] += cross[1]
                    b1[y][x] += (cross[2] + 1) / 2
        r1, g1, b1 = r1 / 8, g1 / 8, b1 / 8
        # print('r')
        # print(r)
        # print(r1)
        # print('g')
        # print(g)
        # print(g1)
        # print('b')
        # print(b)
        # print(b1)
        
        img = Image.fromarray(self.colorize([r1, g1, b1]), mode='RGBA')
        if export:
            img.save(
                '{}.{}.({},{}).png'.format(self.world.name, 'normal', self.x,
                                           self.y), 'PNG')
        return img


class DSTile(Tile):
    def __init__(self, world, x, y, **kwargs):
        if world.getData('func') is None:
            def defaultFunc(array, size, x, y, row, col, r, h, p, rand):
                return min(max(h + (rand - 0.5) * r / p, 0.0), 1.0)
            
            func = kwargs.get('func', None)
            world.data['func'] = func if func else defaultFunc
        
        self.period = kwargs.get('period', 2 ** 5)
        self.func = world.getData('func')
        
        self.corners = [None] * 4
        
        super().__init__(world, x, y, **kwargs)
        
        self.rawArray = np.copy(self.array)
        self.array = self.array[:world.size, :world.size]
    
    def getSeed(self):
        return self.world.seed * self.period
    
    def createArray(self):
        r = 2
        while True:
            if self.world.size <= r: break
            r *= 2
        return np.zeros(shape=(r + 1, r + 1))
    
    def generate(self, freq, amp):
        array = self.createArray() - 1
        newSize = array.shape[0]
        r = newSize - 1
        
        if self.neighbors[1] is not None:
            array[0, :] = self.neighbors[1].rawArray[-1, :]
        if self.neighbors[3] is not None:
            array[:, 0] = self.neighbors[3].rawArray[:, -1]
        if self.neighbors[4] is not None:
            array[:, -1] = self.neighbors[4].rawArray[0, :]
        if self.neighbors[6] is not None:
            array[-1, :] = self.neighbors[6].rawArray[0, :]
        
        if array[0][0] == -1:
            array[0][0] = self.getCornerValue([0, 1, 3], [3, 2, 1])
        if array[0][r] == -1:
            array[0][r] = self.getCornerValue([1, 2, 4], [3, 2, 0])
        if array[r][0] == -1:
            array[r][0] = self.getCornerValue([3, 5, 6], [3, 1, 0])
        if array[r][r] == -1:
            array[r][r] = self.getCornerValue([4, 6, 7], [2, 1, 0])
        
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
        return np.random.random()
    
    def setValue(self, array, row, col, r, h):
        if array[row][col] == -1:
            array[row][col] = self.func(array, self.world.size, self.x, self.y,
                                        row, col, r, h, self.period,
                                        np.random.random())
    
    def diamond(self, array, row, col, r):
        values = [
            array[row - r][col - r],
            array[row - r][col + r],
            array[row + r][col - r],
            array[row + r][col + r]
        ]
        self.setValue(array, row, col, r, sum(values) / len(values))
    
    def square(self, array, row, col, r):
        values = []
        
        if 0 <= row - r:
            values.append(array[row - r][col])
        elif self.neighbors[1] is not None:
            values.append(self.neighbors[1].array[row - r][col])
        if 0 <= col - r:
            values.append(array[row][col - r])
        elif self.neighbors[3] is not None:
            values.append(self.neighbors[3].array[row][col - r])
        if row + r < array.shape[0]:
            values.append(array[row + r][col])
        elif self.neighbors[6] is not None:
            values.append(
                self.neighbors[6].array[row + r - self.world.size][col])
        if col + r < array.shape[1]:
            values.append(array[row][col + r])
        elif self.neighbors[4] is not None:
            values.append(
                self.neighbors[4].array[row][col + r - self.world.size])
        
        self.setValue(array, row, col, r, sum(values) / len(values))


class PerlinTile(Tile):
    def __init__(self, world, x, y, **kwargs):
        step = kwargs.get('step', 45)
        if world.getData('grad') is None:
            world.data['grad'] = np.array(
                [[np.cos(np.radians(ang)), np.sin(np.radians(ang))] for ang in
                 range(0, 360, step)])
        
        self.step = 360 // step
        self.grad = world.getData('grad')
        
        super().__init__(world, x, y, **kwargs)
    
    def getSeed(self):
        return self.world.seed * self.step
    
    def generate(self, freq, amp):
        lin = np.linspace(0, freq, self.world.size, endpoint=False)
        x, y = np.meshgrid(lin, lin)
        
        p = np.zeros(shape=(freq + 1, freq + 1), dtype=int)
        for row in range(0, freq + 1):
            for col in range(0, freq + 1):
                self.setSeed(self.x + (col / freq), self.y + (row / freq))
                p[row][col] = np.random.randint(0, self.step)
        
        xi = x.astype(int)
        yi = y.astype(int)
        
        xf = x - xi
        yf = y - yi
        
        u = self.fade(xf)
        v = self.fade(yf)
        
        n0 = self.gradient(p[yi + 0, xi + 0], xf - 0, yf - 0)
        n1 = self.gradient(p[yi + 0, xi + 1], xf - 1, yf - 0)
        x1 = self.lerp(n0, n1, u)
        
        n0 = self.gradient(p[yi + 1, xi + 0], xf - 0, yf - 1)
        n1 = self.gradient(p[yi + 1, xi + 1], xf - 1, yf - 1)
        x2 = self.lerp(n0, n1, u)
        
        return self.lerp(x1, x2, v) + 0.5
    
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


class WorleyTile(Tile):
    def __init__(self, world, x, y, **kwargs):
        self.count = kwargs.get('count', 4)
        self.correction = kwargs.get('correction', 0.75)
        
        if world.getData('func') is None:
            def defaultFunc(tile, array):
                return array[:, :, 0]
            
            func = kwargs.get('func', None)
            world.data['func'] = func if func else defaultFunc
        
        self.func = world.getData('func')
        
        super().__init__(world, x, y, **kwargs)
    
    def getSeed(self):
        return self.world.seed * self.count
    
    def generate(self, freq, amp):
        points = np.empty((0, 2))
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                self.setSeed(self.x + x, self.y + y)
                p = np.random.random(size=(self.count * freq, 2))
                points = np.vstack((points, np.add(p, [x, y])))
        
        lin = np.linspace(0, 1, self.world.size, endpoint=False)
        x, y = np.meshgrid(lin, lin)
        
        newShape = (self.world.size, self.world.size, 1)
        array = self.createArray().reshape(newShape) + 1
        for p in points:
            xVals = x - p[0]
            yVals = y - p[1]
            dist = np.sqrt(xVals * xVals + yVals * yVals)
            
            array = np.concatenate((array, dist.reshape(newShape)), axis=-1)
        array = np.sort(array, axis=-1)
        
        return self.func(self, array) * np.sqrt(
            self.count * freq) * self.correction


class ValueTile(Tile):
    def __init__(self, world, x, y, **kwargs):
        super().__init__(world, x, y, **kwargs)
    
    def generate(self, freq, amp):
        lin = np.linspace(0, freq, self.world.size, endpoint=False)
        x, y = np.meshgrid(lin, lin)
        
        p = np.zeros(shape=(freq + 1, freq + 1))
        for row in range(0, freq + 1):
            for col in range(0, freq + 1):
                self.setSeed(self.x + (col / freq), self.y + (row / freq))
                p[row][col] = np.random.random()
        
        xi = x.astype(int)
        yi = y.astype(int)
        
        u = self.fade(x - xi)
        v = self.fade(y - yi)
        
        x1 = self.lerp(p[yi, xi], p[yi, xi + 1], u)
        x2 = self.lerp(p[yi + 1, xi], p[yi + 1, xi + 1], u)
        
        return self.lerp(x1, x2, v)
    
    @staticmethod
    def lerp(a, b, x):
        return a + x * (b - a)
    
    @staticmethod
    def fade(t):
        return t * t * (3 - 2 * t)


class OpenSimplexTile(Tile):
    STRETCH_CONSTANT = -0.211324865405187  # (1/Math.sqrt(2+1)-1)/2
    SQUISH_CONSTANT = 0.366025403784439  # (Math.sqrt(2+1)-1)/2
    NORM_CONSTANT = 47 * 2
    GRADIENT = np.array(
        [[5, 2], [2, 5], [-5, 2], [-2, 5], [5, -2], [2, -5], [-5, -2],
         [-2, -5]])
    
    def __init__(self, world, x, y, **kwargs):
        super().__init__(world, x, y, **kwargs)
    
    def generate(self, freq, amp):
        lin = np.linspace(0, freq, self.world.size, endpoint=False)
        x, y = np.meshgrid(lin, lin)
        
        x += self.x * freq
        y += self.y * freq
        
        # Place input coordinates onto grid.
        stretchOffset = (x + y) * OpenSimplexTile.STRETCH_CONSTANT
        xs = x + stretchOffset
        ys = y + stretchOffset
        
        # Floor to get grid coordinates of rhombus (stretched square) super-cell origin.
        xsb = np.floor(xs).astype(int)
        ysb = np.floor(ys).astype(int)
        
        # Skew out to get actual coordinates of rhombus origin. We'll need these later.
        squishOffset = (xsb + ysb) * OpenSimplexTile.SQUISH_CONSTANT
        xb = xsb + squishOffset
        yb = ysb + squishOffset
        
        xRange = xsb.max() - xsb.min() + 3
        yRange = ysb.max() - ysb.min() + 3
        p = np.zeros(shape=(xRange, yRange), dtype=int)
        for j in range(yRange):
            for i in range(xRange):
                col = i + xsb.min() - 1
                row = j + ysb.min() - 1
                self.setSeed(col / freq, row / freq)
                p[j][i] = np.random.randint(0, 8)
        xOffset = xsb.min()
        yOffset = ysb.min()
        
        # Compute grid coordinates relative to rhombus origin.
        xins = xs - xsb
        yins = ys - ysb
        
        # Sum those together to get a value that determines which region we're in.
        inSum = xins + yins
        
        # Positions relative to origin point.
        dx0 = x - xb
        dy0 = y - yb
        
        # We'll be defining these inside the next block and using them afterwards.
        xsv_ext, ysv_ext = self.createArray().astype(
            int), self.createArray().astype(int)
        dx_ext, dy_ext = self.createArray(), self.createArray()
        
        value = self.createArray()
        
        # Contribution (1,0)
        dx1 = dx0 - 1 - OpenSimplexTile.SQUISH_CONSTANT
        dy1 = dy0 - 0 - OpenSimplexTile.SQUISH_CONSTANT
        attn1 = 2 - dx1 * dx1 - dy1 * dy1
        np.multiply(attn1, attn1, out=attn1, where=attn1 > 0)
        np.add(value, attn1 * attn1 * self.extrapolate(
            p[ysb + 0 - yOffset, xsb + 1 - xOffset], dx1, dy1), out=value,
               where=attn1 > 0)
        
        # Contribution (0,1)
        dx2 = dx0 - 0 - OpenSimplexTile.SQUISH_CONSTANT
        dy2 = dy0 - 1 - OpenSimplexTile.SQUISH_CONSTANT
        attn2 = 2 - dx2 * dx2 - dy2 * dy2
        np.multiply(attn2, attn2, out=attn2, where=attn2 > 0)
        np.add(value, attn2 * attn2 * self.extrapolate(
            p[ysb + 1 - yOffset, xsb + 0 - xOffset], dx2, dy2), out=value,
               where=attn2 > 0)
        
        inSumFlag = inSum <= 1  # We're inside the triangle (2-Simplex) at (0,0)
        notInSumFlag = np.logical_not(
            inSumFlag)  # We're inside the triangle (2-Simplex) at (1,1)
        
        zins = self.createArray()
        np.add(1, -inSum, out=zins, where=inSumFlag)
        np.add(2, -inSum, out=zins, where=np.logical_not(inSumFlag))
        
        state1 = np.logical_and(inSumFlag,
                                np.logical_and((zins > xins) + (zins > yins),
                                               xins > yins))  # (0,0) is one of the closest two triangular vertices
        state2 = np.logical_and(inSumFlag,
                                np.logical_and((zins > xins) + (zins > yins),
                                               np.logical_not(xins > yins)))
        state3 = np.logical_and(inSumFlag, np.logical_not((zins > xins) + (
                    zins > yins)))  # (1,0) and (0,1) are the closest two vertices.
        state4 = np.logical_and(notInSumFlag,
                                np.logical_and((zins < xins) + (zins < yins),
                                               xins > yins))  # (0,0) is one of the closest two triangular vertices
        state5 = np.logical_and(notInSumFlag,
                                np.logical_and((zins < xins) + (zins < yins),
                                               np.logical_not(xins > yins)))
        state6 = np.logical_and(notInSumFlag, np.logical_not((zins < xins) + (
                    zins < yins)))  # (1,0) and (0,1) are the closest two vertices.
        
        np.add(xsb, 1, out=xsv_ext, where=state1)
        np.add(ysb, -1, out=ysv_ext, where=state1)
        np.add(dx0, -1, out=dx_ext, where=state1)
        np.add(dy0, 1, out=dy_ext, where=state1)
        
        np.add(xsb, -1, out=xsv_ext, where=state2)
        np.add(ysb, 1, out=ysv_ext, where=state2)
        np.add(dx0, 1, out=dx_ext, where=state2)
        np.add(dy0, -1, out=dy_ext, where=state2)
        
        np.add(xsb, 1, out=xsv_ext, where=state3)
        np.add(ysb, 1, out=ysv_ext, where=state3)
        np.add(dx0, -1 - 2 * OpenSimplexTile.SQUISH_CONSTANT, out=dx_ext,
               where=state3)
        np.add(dy0, -1 - 2 * OpenSimplexTile.SQUISH_CONSTANT, out=dy_ext,
               where=state3)
        
        np.add(xsb, 2, out=xsv_ext, where=state4)
        np.add(ysb, 0, out=ysv_ext, where=state4)
        np.add(dx0, -2 - 2 * OpenSimplexTile.SQUISH_CONSTANT, out=dx_ext,
               where=state4)
        np.add(dy0, -0 - 2 * OpenSimplexTile.SQUISH_CONSTANT, out=dy_ext,
               where=state4)
        
        np.add(xsb, 0, out=xsv_ext, where=state5)
        np.add(ysb, 2, out=ysv_ext, where=state5)
        np.add(dx0, -0 - 2 * OpenSimplexTile.SQUISH_CONSTANT, out=dx_ext,
               where=state5)
        np.add(dy0, -2 - 2 * OpenSimplexTile.SQUISH_CONSTANT, out=dy_ext,
               where=state5)
        
        np.add(xsb, 0, out=xsv_ext, where=state6)
        np.add(ysb, 0, out=ysv_ext, where=state6)
        np.add(dx0, 0, out=dx_ext, where=state6)
        np.add(dy0, 0, out=dy_ext, where=state6)
        
        np.add(xsb, 1, out=xsb, where=notInSumFlag)
        np.add(ysb, 1, out=ysb, where=notInSumFlag)
        np.add(dx0, -1 - 2 * OpenSimplexTile.SQUISH_CONSTANT, out=dx0,
               where=notInSumFlag)
        np.add(dy0, -1 - 2 * OpenSimplexTile.SQUISH_CONSTANT, out=dy0,
               where=notInSumFlag)
        
        # Contribution(0, 0) or (1, 1)
        attn0 = 2 - dx0 * dx0 - dy0 * dy0
        np.multiply(attn0, attn0, out=attn0, where=attn0 > 0)
        np.add(value,
               attn0 * attn0 * self.extrapolate(p[ysb - yOffset, xsb - xOffset],
                                                dx0, dy0), out=value,
               where=attn0 > 0)
        
        # Extra Vertex
        attn_ext = 2 - dx_ext * dx_ext - dy_ext * dy_ext
        np.multiply(attn_ext, attn_ext, out=attn_ext, where=attn_ext > 0)
        np.add(value, attn_ext * attn_ext * self.extrapolate(
            p[ysv_ext - yOffset, xsv_ext - xOffset], dx_ext, dy_ext), out=value,
               where=attn_ext > 0)
        
        return value / OpenSimplexTile.NORM_CONSTANT + 0.5
    
    @staticmethod
    def extrapolate(h, x, y):
        g = OpenSimplexTile.GRADIENT[h]
        return g[:, :, 0] * x + g[:, :, 1] * y


class ModifiedPerlinTile(Tile):
    def __init__(self, world, x, y, **kwargs):
        step = kwargs.get('step', 45)
        if world.getData('grad') is None:
            world.data['grad'] = np.array(
                [[np.cos(np.radians(ang)), np.sin(np.radians(ang))] for ang in
                 range(0, 360, step)])
        
        self.step = 360 // step
        self.grad = world.getData('grad')
        
        super().__init__(world, x, y, **kwargs)
    
    def getSeed(self):
        return self.world.seed * self.step
    
    def generate(self, freq, amp):
        lin = np.linspace(0, freq, self.world.size, endpoint=False)
        x, y = np.meshgrid(lin, lin)
        
        p = np.zeros(shape=(freq + 1, freq + 1), dtype=int)
        for row in range(0, freq + 1):
            for col in range(0, freq + 1):
                self.setSeed((self.x * freq + col) / freq,
                             (self.y * freq + row) / freq)
                p[row][col] = np.random.randint(0, self.step)
        
        pointGroups = []
        gradGroups = []
        for row in range(freq):
            for col in range(freq):
                row1 = row + 1
                col1 = col + 1
                pointGroups.append(
                    [(col, row), (col1, row), (col1, row1), (col, row1)])
                gradGroups.append(
                    [self.grad[p[row][col]], self.grad[p[row][col1]],
                     self.grad[p[row1][col1]], self.grad[p[row1][col]]])
        
        values = self.createArray()
        for i in range(freq ** 2):
            value = self.pointContributions(x, y, pointGroups[i], gradGroups[i])
            np.add(values, value, out=values, where=values == 0)
        
        return values
    
    def pointContributions(self, x, y, points, gradients):
        if len(points) < 2:
            print('Needs at least two points.')
            return False
        
        finalResults = None
        withinPoly = None
        for startIndex, pFirst in enumerate(points):
            groups = []
            pLast = None
            currentIndex = startIndex + 1 if startIndex + 1 < len(points) else 0
            while True:
                p = points[currentIndex]
                if pLast is not None:
                    groups.append((pFirst, pLast, p))
                pLast = p
                
                currentIndex = currentIndex + 1 if currentIndex + 1 < len(
                    points) else 0
                if currentIndex == startIndex:
                    break
            
            results = None
            for group in groups:
                inTriangle = None
                attenuation = None
                for i, p0 in enumerate(group):
                    p1 = group[i - 1]
                    p2 = group[i - 2]
                    
                    p0x, p0y = p0
                    p1x, p1y = p1
                    p2x, p2y = p2
                    
                    dP12x, dP12y = p2x - p1x, p2y - p1y
                    
                    if dP12x == 0:
                        projectedX = p1x
                        projectedY = y
                        projectedP0x = p1x
                        projectedP0y = p0y
                    elif dP12y == 0:
                        projectedX = x
                        projectedY = p1y
                        projectedP0x = p0x
                        projectedP0y = p1y
                    else:
                        projectedX = (x + p1x + (dP12x / dP12y) * (y - p1y)) / 2
                        projectedY = ((dP12y / dP12x) * (x - p1x) + y + p1y) / 2
                        projectedP0x = (p0x + p1x + (dP12x / dP12y) * (
                                    p0y - p1y)) / 2
                        projectedP0y = ((dP12y / dP12x) * (
                                    p0x - p1x) + p0y + p1y) / 2
                    
                    onLine = ((min(projectedP0x, p1x, p2x) <= projectedX) * (
                                projectedX <= max(projectedP0x, p1x, p2x)) *
                              (min(projectedP0y, p1y, p2y) <= projectedY) * (
                                          projectedY <= max(projectedP0y, p1y,
                                                            p2y)))
                    
                    rightSide = (dP12x * (y - projectedY) - dP12y * (
                                x - projectedX)) <= 0
                    
                    if p0 == pFirst:
                        grad = gradients[points.index(p0)]
                        dot = grad[0] * (x - p0x) + grad[1] * (y - p0y)
                        
                        # X Distance
                        a = p2x - projectedP0x
                        b = p2y - projectedP0y
                        c = p1x - projectedP0x
                        d = p1y - projectedP0y
                        maxXDist = max(np.sqrt(a * a + b * b),
                                       np.sqrt(c * c + d * d))
                        
                        a = projectedX - projectedP0x
                        b = projectedY - projectedP0y
                        xDist = self.fade(1 - np.sqrt(a * a + b * b) / maxXDist)
                        
                        # Y Distance
                        a = p0x - projectedP0x
                        b = p0y - projectedP0y
                        maxYDist = np.sqrt(a * a + b * b)
                        
                        a = x - projectedX
                        b = y - projectedY
                        yDist = self.fade(np.sqrt(a * a + b * b) / maxYDist)
                        
                        attenuation = xDist * yDist
                    
                    if inTriangle is None:
                        inTriangle = onLine * rightSide
                    else:
                        inTriangle = inTriangle * onLine * rightSide
                
                if withinPoly is None:
                    withinPoly = inTriangle
                else:
                    withinPoly = withinPoly + inTriangle
                
                if results is None:
                    results = inTriangle * attenuation
                else:
                    np.add(results, inTriangle * attenuation, out=results,
                           where=results == 0)
            
            if finalResults is None:
                finalResults = results * dot
            else:
                finalResults = finalResults + results * dot
        
        return np.add(finalResults, 0.5, where=withinPoly > 0)
    
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


class ModifiedOSTile(Tile):
    def __init__(self, world, x, y, **kwargs):
        step = kwargs.get('step', 45)
        if world.getData('grad') is None:
            world.data['grad'] = np.array(
                [[np.cos(np.radians(ang)), np.sin(np.radians(ang))] for ang in
                 range(0, 360, step)])
        
        self.step = 360 // step
        self.grad = world.getData('grad')
        
        super().__init__(world, x, y, **kwargs)
    
    def getSeed(self):
        return self.world.seed * self.step
    
    def generate(self, freq, amp):
        freq = 1
        lin = np.linspace(0, freq, self.world.size, endpoint=False)
        x, y = np.meshgrid(lin, lin)
        
        p = np.zeros(shape=(2 * freq + 3, (freq + 1) * 3), dtype=int)
        for j in range(0, 2 * freq + 3):
            for i in range(0, (freq + 1) * 3):
                row = round((j - 1) / 2, 9)
                col = round((i - 1) / 3, 9)
                self.setSeed((self.x * freq + col) / freq,
                             (self.y * freq + row) / freq)
                p[j][i] = np.random.randint(0, self.step)
        
        a, b = 1.5, -0.5
        
        workX, workY = a * x + b * y, b * x + a * y
        workSum = workX + workY
        
        negY = np.logical_and(workSum < 1, workY < 0)
        negX = np.logical_and(workSum < 1, workX < 0)
        posX = np.logical_and(1 <= workSum, 1 <= workX)
        posY = np.logical_and(1 <= workSum, 1 <= workY)
        cent = np.logical_not(negY + negX + posX + posY)
        
        value = self.createArray()
        
        # Neg Y
        offset = (-0.25, 0.25)
        transform = (1, 1,
                     -1, 3)
        grads = (p[0][2], p[1][4],
                 p[1][1], p[2][3])
        print(grads)
        mask = negY
        value += self.calcSection(x, y, offset, transform, grads, mask)
        
        # Neg X
        offset = (0.25, -0.25)
        transform = (3, -1,
                     1, 1)
        grads = (p[2][0], p[1][1],
                 p[3][1], p[2][2])
        print(grads)
        mask = negX
        value += self.calcSection(x, y, offset, transform, grads, mask)
        
        # Pos X
        offset = (-0.75, -0.25)
        transform = (3, -1,
                     1, 1)
        grads = (p[2][3], p[1][4],
                 p[3][4], p[2][5])
        print(grads)
        mask = posX
        value += self.calcSection(x, y, offset, transform, grads, mask)
        
        # Pos Y
        offset = (-0.25, -0.75)
        transform = (1, 1,
                     -1, 3)
        grads = (p[2][2], p[3][4],
                 p[3][1], p[4][3])
        print(grads)
        mask = posY
        value += self.calcSection(x, y, offset, transform, grads, mask)
        
        # Center
        offset = (0, 0)
        transform = (1.5, -0.5,
                     -0.5, 1.5)
        grads = (p[1][1], p[2][3],
                 p[2][2], p[3][4])
        print(grads)
        mask = cent
        value += self.calcSection(x, y, offset, transform, grads, mask)
        
        # print(value)
        
        return value
    
    def calcSection(self, x, y, offset, transform, grads, mask):
        workX = x + offset[0]
        workY = y + offset[1]
        
        workX, workY = transform[0] * workX + transform[1] * workY, transform[
            2] * workX + transform[3] * workY
        
        u = self.fade(workX)
        v = self.fade(workY)
        
        n0 = self.gradient(self.grad[grads[0]], workX, workY)
        n1 = self.gradient(self.grad[grads[1]], workX - 1, workY)
        x1 = self.lerp(n0, n1, u)
        
        n0 = self.gradient(self.grad[grads[2]], workX, workY - 1)
        n1 = self.gradient(self.grad[grads[3]], workX - 1, workY - 1)
        x2 = self.lerp(n0, n1, u)
        
        return (self.lerp(x1, x2, v) + 0.5) * mask
    
    @staticmethod
    def gradient(g, x, y):
        return g[0] * x + g[1] * y
    
    @staticmethod
    def lerp(a, b, x):
        return a + x * (b - a)
    
    @staticmethod
    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)


class NoiseArray:
    def __init__(self, dimension, **kwargs):
        self.dimension = dimension
        
        self.seed = kwargs.get('seed', 1)
        self.octaves = kwargs.get('octaves', 1)
        self.persistence = kwargs.get('persistence', 0.5)
    
    def validateInput(self, axis):
        if len(axis) != self.dimension:
            raise Exception(
                'Inputed Coordinate Dimension \'{}\' does not match Dimension Specified \'{}\''.format(
                    len(axis), self.dimension))
    
    def getSeed(self):
        return self.seed
    
    def setSeed(self, *coord):
        h = 0
        for c in '{}:{}'.format(','.join(map(str, coord)), self.getSeed()):
            h = (31 * h + 17 * ord(c)) & 0xFFFFFFFF
        np.random.seed(h)
    
    def calculateRange(self, *ranges):
        linSpaces = []
        for rng in ranges:
            start = rng[0] if type(rng) == list else 0
            end = rng[1] if type(rng) == list else rng
            step = rng[2] if type(rng) == list and len(rng) > 2 else 1
            linSpaces.append(np.linspace(start, end, int((end - start) / step),
                                         endpoint=False))
        return self.calculateValues(*np.meshgrid(*linSpaces))
    
    def calculateValues(self, *coord):
        self.validateInput(coord)
        
        value = np.copy(coord) * 0
        
        maxValue = 0
        amp = 1
        freq = 1
        for i in range(self.octaves):
            value += self.generate(np.array(coord), freq, amp) * amp
            maxValue += amp
            amp *= self.persistence
            freq *= 2
        value /= maxValue
        
        return np.clip(value, 0, 1)
    
    def generate(self, coords, freq, amp):
        return 0
    
    @staticmethod
    def floor(x):
        i = np.int64(x)
        return i - (i > x)
    
    @staticmethod
    def ceiling(x):
        i = np.int64(x)
        return i - (i > x) + 1
    
    @staticmethod
    def lerp(a, b, x):
        return a + x * (b - a)
    
    @staticmethod
    def fade(t):
        """6t^5 - 15t^4 + 10t^3"""
        return t * t * t * (t * (t * 6 - 15) + 10)


class PerlinArray(NoiseArray):
    def __init__(self, dimension, **kwargs):
        super().__init__(dimension, **kwargs)
        
        self.grad = np.array([
            [j if k == i else 0 for k in range(self.dimension)]
            for i in range(self.dimension)
            for j in [1, -1]
        ], dtype=int)
    
    def generate(self, coords, freq, amp):
        values = coords * freq
        
        extrema = np.array([
            [self.floor(values[i].min()), self.ceiling(values[i].max())]
            for i in range(self.dimension)
        ])
        
        p = np.zeros(shape=(extrema[:, 1] - extrema[:, 0] + 1)[::-1],
                     dtype='uint8')
        for index, _ in np.ndenumerate(p):
            self.setSeed(extrema[:, 0] + index[::-1])
            p[index] = np.random.randint(0, 2 * self.dimension)
        
        valuesI = self.floor(values)
        
        valuesF = values - valuesI
        
        fade = self.fade(valuesF)
        
        gradients = np.empty((2 ** self.dimension, *values.shape[1:]))
        for i in range(2 ** self.dimension):
            offset = np.array(
                [int(x) for x in format(i, '0{}b'.format(self.dimension))])[
                     ::-1]
            
            node = (valuesI.T + offset - extrema[:, 0]).T
            direction = (valuesF.T - offset).T
            
            nodeDirections = self.grad[
                p[tuple(np.split(node[::-1], self.dimension))][0]]
            nodeDirections = np.reshape(
                np.split(nodeDirections, self.dimension, axis=-1), values.shape)
            
            gradients[i] = np.sum(nodeDirections * direction, axis=0)
        
        for i in range(self.dimension):
            for j in range(gradients.shape[0] // 2):
                gradients[j] = self.lerp(gradients[2 * j], gradients[2 * j + 1],
                                         fade[i])
            gradients = gradients[:gradients.shape[0] // 2]
        gradients = np.reshape(gradients, gradients.shape[1:]) + 0.5
        
        return gradients


class WorleyArray(NoiseArray):
    def __init__(self, dimension, **kwargs):
        super().__init__(dimension, **kwargs)
        
        self.count = kwargs.get('count', 4)
        self.correction = kwargs.get('correction', 0.75)
        # Array is distances to points [0, number of points)
        func = kwargs.get('func', None)
        self.func = func if func else (lambda array: array[0])
    
    def getSeed(self):
        return self.seed * self.count
    
    def generate(self, coords, freq, amp):
        values = coords.copy()
        
        # We subtract 1 from the minimum to allow for points to be populated to the top and left
        extrema = np.array([
            [self.floor(values[i].min()) - 1, self.ceiling(values[i].max())]
            for i in range(self.dimension)
        ])
        
        points = np.empty((0, self.dimension))
        p = np.empty(shape=(extrema[:, 1] - extrema[:, 0] + 1)[::-1])
        for index, _ in np.ndenumerate(p):
            index = extrema[:, 0] + index[::-1]
            self.setSeed(index)
            p = np.random.random(size=(self.count * freq, self.dimension))
            points = np.vstack((points, index + p))
        
        distanceArray = np.ones(shape=(0, *coords.shape[1:])) * 3
        for p in points:
            axialDif = (values.T - p).T
            dist = np.sqrt(np.sum(axialDif * axialDif, axis=0))
            
            distanceArray = np.vstack(
                (distanceArray, dist.reshape((1, *coords.shape[1:]))))
        
        distanceArray = np.sort(distanceArray, axis=0)
        
        return self.func(distanceArray) * np.sqrt(
            self.count * freq) * self.correction


class ValueArray(NoiseArray):
    def generate(self, coords, freq, amp):
        values = coords * freq
        
        extrema = np.array([
            [self.floor(values[i].min()), self.ceiling(values[i].max())]
            for i in range(self.dimension)
        ])
        
        p = np.zeros(shape=(extrema[:, 1] - extrema[:, 0] + 1)[::-1])
        for index, _ in np.ndenumerate(p):
            self.setSeed(extrema[:, 0] + index[::-1])
            p[index] = np.random.random()
        
        valuesI = self.floor(values)
        
        fade = self.fade(values - valuesI)
        
        gradients = np.empty((2 ** self.dimension, *values.shape[1:]))
        for i in range(2 ** self.dimension):
            offset = np.array(
                [int(x) for x in format(i, '0{}b'.format(self.dimension))])[
                     ::-1]
            
            node = (valuesI.T + offset - extrema[:, 0]).T
            
            gradients[i] = p[tuple(np.split(node[::-1], self.dimension))][0]
        
        for i in range(self.dimension):
            for j in range(gradients.shape[0] // 2):
                gradients[j] = self.lerp(gradients[2 * j], gradients[2 * j + 1],
                                         fade[i])
            gradients = gradients[:gradients.shape[0] // 2]
        gradients = np.reshape(gradients, gradients.shape[1:])
        
        return gradients


if __name__ == '__main__':
    import os
    import sys
    
    currentDir = os.path.dirname(os.path.realpath(__file__))
    if currentDir not in sys.path: sys.path.append(currentDir)
    os.chdir(currentDir)
    
    np.set_printoptions(linewidth=9999, precision=3, edgeitems=10,
                        threshold=8000, suppress=True)
    
    
    def debugImage(name, array, show=False, sections=None, scale=False):
        if type(array) != np.ndarray:
            array = np.array([[array]])
        if show:
            print(name)
            print(array)
        array = array * 1
        if scale and array.min() != array.max():
            array -= array.min()
            max = array.max()
            array = array / max if max != 0 else array
        array *= 255
        if sections is not None:
            sections = 256 / sections
            array = array // sections * sections
        img = Image.fromarray(array.astype('uint8'))
        img.save('{}.png'.format(name), 'PNG')
    
    
    def fade(t):
        """6t^5 - 15t^4 + 10t^3"""
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    
    dimension = 1
    seed = 1
    octaves = 1
    persistence = 0.9
    
    worleyFunc = lambda array: array[1] - array[0]
    worleyFunc = lambda array: 1 - ((1 - array[0]) * (1 - array[0]))
    worleyFunc = lambda array: fade(array[0])
    worleyFunc = None
    
    arrayTest = PerlinArray(dimension, seed=seed, octaves=octaves,
                            persistence=persistence)
    # arrayTest = WorleyArray(dimension, seed = seed, octaves = octaves, persistence = persistence, func = worleyFunc)
    # arrayTest = ValueArray(dimension, seed = seed, octaves = octaves, persistence = persistence)
    
    mn, mx, inc = 10, 11, 1 / 512
    x = [mn, mx, inc]
    y = [mn, mx, inc]
    z = [mn, mx, inc]
    w = [mn, mx, inc]
    
    for i in range(1):
        values = arrayTest.calculateRange(x)
        # values = arrayTest.calculateRange(x, y)
        # values = arrayTest.calculateRange(x, y, z)
        # values = arrayTest.calculateRange(x, y, z, w)
        print(values)
        
        if dimension == 2:
            debugImage('out/Test {}'.format(i + 1), values, show=False)
        
        x[0] += 0.25
        x[1] += 0.25
    
    pass
