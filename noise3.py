# System Packages

# My Packages
import Util
# Third-Party Packages
import numpy as np
from PIL import Image


def floor(x):
    i = np.int64(x)
    return i - (i > x)


def ceiling(x):
    i = np.int64(x)
    return i - (i > x) + 1


def lerp(a, b, x):
    return a + x * (b - a)


def fade(t):
    """6t^5 - 15t^4 + 10t^3"""
    return t * t * t * (t * (t * 6 - 15) + 10)


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
def to_rgb(arr, tint=None, sections=None):
    arr = np.uint8((arr * 255) if _is_float(arr) else arr)
    color = np.zeros((*arr.shape, 3), dtype=np.uint8)
    color[:, :, 0] = color[:, :, 1] = color[:, :, 2] = arr
    if tint is not None:
        tint = np.array(tint)
        color = np.uint8((color[:, :] / 255 * tint / 255) * 255)
    if sections is not None and sections > 0:
        sections = 256 / sections
        color = np.uint8(color // sections * sections)
    return color


_neighbors = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1),
              (-1, 0)]


def to_normal(arr):
    lins = [np.linspace(0, dim, dim, endpoint=False, dtype=int) for dim in
            arr.shape]
    x, y = np.meshgrid(*lins)
    
    neighbors = []
    for i, j in _neighbors:
        x_pos = x + i
        y_pos = y + i
        
        x_dir = np.zeros_like(x) + i / x.shape[1]
        y_dir = np.zeros_like(y) + j / y.shape[0]
        
        np.add(x, 0, out=x_pos, where=(x_pos < 0) + (x.shape[0] <= x_pos))
        np.add(y, 0, out=y_pos, where=(y_pos < 0) + (y.shape[1] <= y_pos))
        
        neighbors.append(np.array([x_dir, y_dir, arr[y_pos, x_pos] - arr]))
    neighbors = np.moveaxis(np.array(neighbors), 1, 3)
    
    normal = np.zeros((*arr.shape, 3))
    for i, p2 in enumerate(neighbors):
        p1 = neighbors[i - 1]
        
        cross = np.cross(p1[y, x], p2[y, x])
        cross = (cross / np.linalg.norm(cross, axis=2).reshape(
            (*arr.shape, 1)) + 1) / 2
        
        normal[:, :, 0] += cross[:, :, 0]
        normal[:, :, 1] += cross[:, :, 1]
        normal[:, :, 2] += (cross[:, :, 2] + 1) / 2
    
    return np.uint8(normal / 8 * 255)


class Noise:
    def __init__(self, *, seed=1, octaves=1, persistence=0.5):
        self.base_seed = seed
        self.octaves = octaves
        self.persistence = persistence
    
    def _get_seed(self):
        return self.base_seed
    
    def _set_seed(self, coord):
        try:
            coord_str = map(str, coord)
        except TypeError:
            coord_str = map(str, (coord,))
        h = 0
        for c in '{}:{}'.format(','.join(coord_str), self.seed):
            h = (31 * h + 17 * ord(c)) & 0xFFFFFFFF
        np.random.seed(h)
    
    seed = property(_get_seed, _set_seed)
    
    def generate_range(self, *ranges):
        lin_spaces = []
        for rng in ranges:
            if type(rng) in (list, tuple):
                start = rng[0]
                end = rng[1]
                step = rng[2] if len(rng) > 2 else 10
            else:
                start = rng
                end = rng
                step = 1
            
            lin_spaces.append(
                np.linspace(
                    start,
                    end,
                    step,
                    endpoint=False
                )
            )
        return self.generate(*np.meshgrid(*lin_spaces))
    
    def generate(self, *coords):
        dimension = len(coords)
        values = np.sum(coords) * 0
        coords = np.array(coords)
        
        self._setup(dimension)
        
        maxValues = 0
        amp = 1
        freq = 1
        for i in range(self.octaves):
            values += self._calculate(dimension, coords, freq, amp) * amp
            maxValues += amp
            amp *= self.persistence
            freq *= 2
        values /= maxValues
        
        return np.clip(values, 0, 1)
    
    def _setup(self, dimension):
        pass
    
    def _calculate(self, coords, freq, amp):
        return 0


class PerlinNoise(Noise):
    grad = None
    
    def _setup(self, dimension):
        self.grad = np.array([
            [j if k == i else 0 for k in range(dimension)]
            for i in range(dimension)
            for j in [1, -1]
        ], dtype=int)
    
    def _calculate(self, dimension, coords, freq, amp):
        values = coords * freq
        
        extrema = np.array([
            [floor(values[i].min()), ceiling(values[i].max())]
            for i in range(dimension)
        ])
        
        p = np.zeros(shape=(extrema[:, 1] - extrema[:, 0] + 1)[::-1],
                     dtype='uint8')
        for index, _ in np.ndenumerate(p):
            self.seed = extrema[:, 0] + index[::-1]
            p[index] = np.random.randint(0, 2 * dimension)
        
        valuesI = floor(values)
        
        valuesF = values - valuesI
        
        _fade = fade(valuesF)
        
        grads = np.empty((2 ** dimension, *values.shape[1:]))
        for i in range(2 ** dimension):
            offset = np.array(
                [int(x) for x in format(i, '0{}b'.format(dimension))])[::-1]
            
            node = (valuesI.T + offset - extrema[:, 0]).T
            dir = (valuesF.T - offset).T
            
            dirs = self.grad[p[tuple(np.split(node[::-1], dimension))][0]]
            dirs = np.reshape(np.split(dirs, dimension, axis=-1), values.shape)
            
            grads[i] = np.sum(dirs * dir, axis=0)
        
        for i in range(dimension):
            for j in range(grads.shape[0] // 2):
                grads[j] = lerp(grads[2 * j], grads[2 * j + 1], _fade[i])
            grads = grads[:grads.shape[0] // 2]
        grads = np.reshape(grads, grads.shape[1:]) + 0.5
        
        return grads


class WorleyNoise(Noise):
    def __init__(self, *,
                 seed=1,
                 octaves=1,
                 persistence=0.5,
                 count=4,
                 correction=0.75,
                 func=None
                 ):
        super().__init__(
            seed=seed,
            octaves=octaves,
            persistence=persistence
        )
        
        self.count = count
        self.correction = correction
        self.func = func if func else (lambda a: a[0])
    
    def _get_seed(self):
        return self.base_seed * self.count
    
    def _set_seed(self, coord):
        super()._set_seed(coord)
    
    seed = property(_get_seed, _set_seed)
    
    def _calculate(self, dimension, coords, freq, amp):
        values = coords.copy()
        
        extrema = np.array([
            [floor(values[i].min()) - 1, ceiling(values[i].max()) + 1]
            for i in range(dimension)
        ])
        
        points = np.empty((0, dimension))
        p = np.empty(shape=(extrema[:, 1] - extrema[:, 0] + 1)[::-1])
        for index, _ in np.ndenumerate(p):
            index = extrema[:, 0] + index[::-1]
            self.seed = index
            p = np.random.random(size=(self.count * freq, dimension))
            points = np.vstack((points, index + p))
        
        dists = np.ones(shape=(0, *coords.shape[1:])) * 3
        for p in points:
            axialDif = (values.T - p).T
            dist = np.sqrt(np.sum(axialDif * axialDif, axis=0))
            
            dists = np.vstack((dists, dist.reshape((1, *coords.shape[1:]))))
        
        dists = np.sort(dists, axis=0)
        
        return self.func(dists) * np.sqrt(self.count * freq) * self.correction


class ValueNoise(Noise):
    def _calculate(self, dimension, coords, freq, amp):
        values = coords * freq
        
        extrema = np.array([
            [floor(values[i].min()), ceiling(values[i].max())]
            for i in range(dimension)
        ])
        
        p = np.zeros(shape=(extrema[:, 1] - extrema[:, 0] + 1)[::-1])
        for index, _ in np.ndenumerate(p):
            self.seed = extrema[:, 0] + index[::-1]
            p[index] = np.random.random()
        
        valuesI = floor(values)
        
        _fade = fade(values - valuesI)
        
        grads = np.empty((2 ** dimension, *values.shape[1:]))
        for i in range(2 ** dimension):
            offset = np.array(
                [int(x) for x in format(i, '0{}b'.format(dimension))]
            )[::-1]
            
            node = (valuesI.T + offset - extrema[:, 0]).T
            
            grads[i] = p[tuple(np.split(node[::-1], dimension))][0]
        
        for i in range(dimension):
            for j in range(grads.shape[0] // 2):
                grads[j] = lerp(grads[2 * j], grads[2 * j + 1], _fade[i])
            grads = grads[:grads.shape[0] // 2]
        grads = np.reshape(grads, grads.shape[1:])
        
        return grads


if __name__ == '__main__':
    Util.setup_env(__file__)
    
    np.set_printoptions(
        linewidth=9999,
        precision=6,
        edgeitems=10,
        threshold=9999,
        suppress=True
    )
    
    noise = WorleyNoise(seed=69, octaves=1)
    
    array = noise.generate_range((0, 2, 400), (0, 2, 400))
    
    normal = to_normal(array)
    print(normal)
    print(normal.shape)
    # normal = to_rgb(normal)
    
    array = to_rgb(array, tint=(100, 200, 50), sections=16)
    
    im = Image.fromarray(array, mode='RGB')
    im.save('test.png', 'PNG')
    im = Image.fromarray(normal, mode='RGB')
    im.save('normal.png', 'PNG')
