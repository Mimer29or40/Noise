# System Packages
import time

# Third-Party Packages
import numpy as np
from PIL import Image


# My Packages

# Project Packages


def get_time(start=None):
    current = time.perf_counter_ns()
    if start is None: return current
    return current - start


def clamp(x, lower, upper):
    return min(max(x, lower), upper)


_ints = [int] + np.sctypes['int'] + np.sctypes['uint']


def _is_int(x):
    return type(x) in _ints or (
            type(x) == np.ndarray and x.dtype in _ints
    )


_floats = [float] + np.sctypes['float']


def _is_float(x):
    return type(x) in _floats or (
            type(x) == np.ndarray and x.dtype in _floats
    )


# Assumes 2D array
def to_rgb(array, tint=None, sections=None):
    array = [array[0]] * 3 if len(array) == 1 else array
    color = [np.uint8((c * 255) if _is_float(c) else c) for c in array]
    if tint is not None:
        tint = to_rgb(tint)
        color = [
            np.uint8((c / 255 * t / 255) * 255)
            for c, t in zip(color, tint)
        ]
    if sections is not None:
        sections = 256 / sections
        color = [np.uint8(c // sections * sections) for c in color]
    return color


def to_color_int(array):
    return (
            (np.uint32(array[0]) << 0)
            + (np.uint32(array[1]) << 8)
            + (np.uint32(array[2]) << 16)
            + (np.uint32(255) << 24)
    )


class World:
    def __init__(self, tile, *, size=64, seed=1, func=None):
        self.tile = tile
        
        self.size = size
        self.seed = seed
        self.func = func
        
        self._tiles = {}
        self._data = {}
    
    def __getitem__(self, key):
        if type(key) is str:
            try:
                return self._data[key]
            except KeyError:
                return None
        try:
            return self._tiles['{},{}'.format(*key)]
        except KeyError:
            return None
    
    def __setitem__(self, key, value):
        if type(key) is str:
            self._data[key] = value
    
    def world_func(self, x, y, array):
        if self.func is not None:
            return np.clip(self.func(x, y, array), 0, 1)
        return array
    
    def get_neighbors(self, x, y):
        return [
            self[x + i, y + j]
            for j in [-1, 0, 1]
            for i in [-1, 0, 1]
            if i != 0 or j != 0
        ]
    
    def gen_tile(self, x, y):
        self._tiles['{},{}'.format(x, y)] = self.tile(self, x, y)
    
    def gen_radius(self, r, **kwargs):
        self._data.update(kwargs)
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                self.gen_tile(i, j)
    
    def to_image(self, name, tint=None, sections=None, debug=False):
        minX, maxX, minY, maxY = self._bounds()
        
        box = ((maxX - minX + 1) * self.size, (maxY - minY + 1) * self.size)
        stitched = Image.new('RGB', box)
        
        for key, tile in self._tiles.items():
            x, y = map(int, key.split(','))
            array = to_color_int(to_rgb([tile.array], tint, sections))
            box = ((x - minX) * self.size, (y - minY) * self.size)
            im = Image.fromarray(array, mode='RGBA')
            if debug:
                im.save('{}.{}.{}.png'.format(name, x, y), 'PNG')
            stitched.paste(im=im, box=box)
        
        stitched.save(name + '.png', 'PNG')
    
    def _bounds(self):
        minX, maxX, minY, maxY = None, None, None, None
        for key in self._tiles.keys():
            x, y = map(int, key.split(','))
            minX = x if minX is None else min(x, minX)
            maxX = x if maxX is None else max(x, maxX)
            minY = y if minY is None else min(y, minY)
            maxY = y if maxY is None else max(y, maxY)
        return minX, maxX, minY, maxY


class Tile:
    def __init__(self, world, x, y):
        self.world = world
        
        self.x = x
        self.y = y
        
        self.set_seed(x, y)
        
        self._neighbors = world.get_neighbors(x, y)
        
        self._array = self.create_array()
        
        t = get_time()
        maxValue = 0
        amp = 1
        freq = 1
        for _ in range(world['octaves']):
            self._array += self.generate(freq, amp) * amp
            maxValue += amp
            amp *= world['persistence']
            freq *= 2
        self._array = np.clip(self._array / maxValue, 0, 1)
        print('Gen Time for ({:2},{:2}): {:>8} ns'.format(x, y, get_time(t)))
    
    @property
    def array(self):
        return self.world.world_func(self.x, self.y, self._array)
    
    @property
    def size(self):
        return self.world.size
    
    @property
    def seed(self):
        return self.world.seed
    
    def set_seed(self, *coord):
        h = 0
        for c in '{}:{}'.format(','.join(map(str, coord)), self.seed):
            h = (31 * h + 17 * ord(c)) & 0xFFFFFFFF
        np.random.seed(h)
    
    def create_array(self):
        return np.zeros(shape=(self.size, self.size), dtype=float)
    
    def generate(self, freq, amp):
        return np.zeros(shape=(self.size, self.size), dtype=float)


class DSTile(Tile):
    def __init__(self, world, x, y):
        self.period = world['period']
        self.func = world['func']
        
        super().__init__(world, x, y)
        
        self._raw = np.copy(self._array)
        self._array = self._array[:self.size, :self.size]
    
    @property
    def seed(self):
        return self.world.seed * self.period
    
    def create_array(self):
        r = 2
        while True:
            if self.size <= r: break
            r *= 2
        return np.zeros(shape=(r + 1, r + 1), dtype=float)
    
    def generate(self, freq, amp):
        array = self.create_array() - 1
        new_size = array.shape[0]
        r = new_size - 1
        
        if self._neighbors[1] is not None:
            array[0, :] = self._neighbors[1]._raw[-1, :]
        if self._neighbors[3] is not None:
            array[:, 0] = self._neighbors[3]._raw[:, -1]
        if self._neighbors[4] is not None:
            array[:, -1] = self._neighbors[4]._raw[0, :]
        if self._neighbors[6] is not None:
            array[-1, :] = self._neighbors[6]._raw[0, :]
        
        if array[0, 0] == -1:
            array[0, 0] = self.corner_value([0, 1, 3], [3, 2, 1])
        if array[0, r] == -1:
            array[0, r] = self.corner_value([1, 2, 4], [3, 2, 0])
        if array[r, 0] == -1:
            array[r, 0] = self.corner_value([3, 5, 6], [3, 1, 0])
        if array[r, r] == -1:
            array[r, r] = self.corner_value([4, 6, 7], [2, 1, 0])
        
        r //= 2
        while r > 0:
            for row in range(r, new_size, 2 * r):
                for col in range(r, new_size, 2 * r):
                    self.diamond(array, row, col, r)
            switch = False
            for row in range(0, new_size, r):
                for col in range(0, new_size, r):
                    if switch:
                        self.square(array, row, col, r)
                    switch = not switch
            r //= 2
        
        return array
    
    def corner(self, x):
        if x == 0: return self._raw[0, 0]
        elif x == 1: return self._raw[0, self.size]
        elif x == 2: return self._raw[self.size, 0]
        elif x == 3: return self._raw[self.size, self.size]
    
    def corner_value(self, neighbors, corners):
        try:
            for c, n in zip(neighbors, corners):
                return self._neighbors[n].corner(c)
        except AttributeError:
            return np.random.random()
    
    @staticmethod
    def default_func(array, size, x, y, row, col, r, h, p, rand):
        return clamp(h + (rand - 0.5) * r / p, 0.0, 1.0)
    
    def set_value(self, array, row, col, r, h):
        args = (
            array, self.world.size, self.x, self.y,
            row, col, r, h, self.period, np.random.random()
        )
        if array[row, col] == -1:
            if self.func is None:
                array[row, col] = self.default_func(*args)
            else:
                array[row, col] = self.func(*args)
    
    def diamond(self, array, row, col, r):
        values = [
            array[row - r, col - r],
            array[row - r, col + r],
            array[row + r, col - r],
            array[row + r, col + r]
        ]
        self.set_value(array, row, col, r, sum(values) / len(values))
    
    def square(self, array, row, col, r):
        values = []
        
        if 0 <= row - r:
            values.append(array[row - r, col])
        elif self._neighbors[1] is not None:
            values.append(self._neighbors[1]._array[row - r, col])
        if 0 <= col - r:
            values.append(array[row, col - r])
        elif self._neighbors[3] is not None:
            values.append(self._neighbors[3]._array[row, col - r])
        if row + r < array.shape[0]:
            values.append(array[row + r, col])
        elif self._neighbors[6] is not None:
            values.append(self._neighbors[6]._array[row + r - self.size, col])
        if col + r < array.shape[1]:
            values.append(array[row, col + r])
        elif self._neighbors[4] is not None:
            values.append(self._neighbors[4]._array[row, col + r - self.size])
        
        self.set_value(array, row, col, r, sum(values) / len(values))


class ValueTile(Tile):
    def generate(self, freq, amp):
        lin = np.linspace(0, freq, self.size, endpoint=False)
        x, y = np.meshgrid(lin, lin)
        
        p = np.zeros(shape=(freq + 1, freq + 1))
        for row in range(0, freq + 1):
            for col in range(0, freq + 1):
                self.set_seed(self.x + (col / freq), self.y + (row / freq))
                p[row, col] = np.random.random()
        
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


class Noise:
    def __init__(self, dimension, seed=1, octaves=1, persistence=0.5):
        self.dimension = dimension
        
        self._seed = seed
        self.octaves = octaves
        self.persistence = persistence
    
    @property
    def seed(self):
        return self._seed
    
    def set_seed(self, *coord):
        h = 0
        for c in '{}:{}'.format(','.join(map(str, coord)), self.seed):
            h = (31 * h + 17 * ord(c)) & 0xFFFFFFFF
        np.random.seed(h)
    
    def calculate(self, *coord):
        if len(axis) != self.dimension:
            raise Exception(
                'Inputed Coordinate Dimension \'{}\' does not match Dimension Specified \'{}\''.format(
                    len(axis), self.dimension))
        
        values = np.sum(coord) * 0
        
        maxValues = 0
        amp = 1
        freq = 1
        for i in range(self.octaves):
            values += self.generate(np.array(coord), freq, amp) * amp
            maxValues += amp
            amp *= self.persistence
            freq *= 2
        values /= maxValues
        
        return np.clip(values, 0, 1)
    
    def generate(self, coords, freq, amp):
        return 0


if __name__ == '__main__':
    import sys
    
    sys.exit()
    Util.setup_env(__file__)
    
    np.set_printoptions(linewidth=9999, precision=6, edgeitems=10,
                        threshold=4000, suppress=True)
    
    
    # World #
    def world_func(x, y, array):
        return np.abs(2 * array - 1)
    
    
    def world_func(x, y, array):
        array *= 4
        return array - array.astype(int)
    
    
    # def world_func(x, y, array):
    # lin = np.linspace(0, 1, array.shape[0], endpoint = False)
    # xVal, yVal = np.meshgrid(lin, lin)
    # xVal += x
    # yVal = perlinNoise.calculateValues((yVal + y + 10) / 10)
    # return ((np.sin(xVal / 2 + np.math.pi * np.cos(yVal / 3) / 1.5) + 1) / 2 + array) / 2
    # return ((np.sin(xVal / 1.5 + yVal * 8) + 1) / 2 + array) / 2
    world_func = None
    
    # Tile
    octaves = 1
    persistence = 0.5
    
    # DS Tile
    period = 2 ** 5
    
    
    def dsFunc(array, size, x, y, row, col, r, h, p, rand):
        xVal = x * size + col - size / 2
        yVal = y * size + row - size / 2
        dist = np.sqrt(
            xVal * xVal + yVal * yVal) if xVal != 0 or yVal != 0 else 0.0000001
        val = h + (rand - 0.5) * r / p * size / dist
        return min(max(val, 0.0), 1.0)
    
    
    # dsFunc = None
    
    world = World(DSTile, func=world_func)
    
    world.gen_radius(16, octaves=octaves, persistence=persistence,
                     period=period, func=dsFunc)
    
    world.to_image('test', sections=None, debug=False)
