import os
import sys

currentDir = os.path.dirname(os.path.realpath(__file__))
if currentDir not in sys.path: sys.path.append(currentDir)
os.chdir(currentDir)

import numpy as np
from PIL import Image

import noise as ns

np.set_printoptions(linewidth=9999, precision=6, edgeitems=10, threshold=4000,
                    suppress=True)


def debugImage(name, array, show=False, sections=None):
    if type(array) != np.ndarray:
        array = np.array([[array]])
    if show:
        print(name)
        print(array)
    array = array * 1
    if array.min() != array.max():
        array -= array.min()
        max = array.max()
        array = array / max if max != 0 else array
    array *= 255
    if sections is not None:
        sections = 256 / sections
        array = array // sections * sections
    img = Image.fromarray(array.astype('uint8'))
    img.save('{}.png'.format(name), 'PNG')


def inPolyTest():
    def inPoly(x, y, points, gradients):
        if len(points) < 2:
            print('Needs at least two points.')
            return False
        
        fade = lambda t: t * t * t * (t * (t * 6 - 15) + 10)
        
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
                        grad = gradient[points.index(p0)]
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
                        xDist = fade(1 - np.sqrt(a * a + b * b) / maxXDist)
                        
                        # Y Distance
                        a = p0x - projectedP0x
                        b = p0y - projectedP0y
                        maxYDist = np.sqrt(a * a + b * b)
                        
                        a = x - projectedX
                        b = y - projectedY
                        yDist = fade(np.sqrt(a * a + b * b) / maxYDist)
                        
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
    
    size = 256
    
    lin = np.linspace(0, 1, size, endpoint=False)
    x, y = np.meshgrid(lin, lin)
    
    points = [(0.25, 0.1), (1., 0.), (1., 1.), (0, 1)]
    # points = [(0., 0.), (1., 0.), (1., 1.), (0, 1)]
    # points = [(0., 0.), (1., 0.), (1., 1.), (0, 1)]
    # points = [(0, 0), (1., 0.), (1., 1.)]
    # points = [(0, 0), (0.5, 0.25), (1, 0), (1, 1)]
    # points = [(np.cos(np.radians(ang)) / 2 + 0.5, np.sin(np.radians(ang)) / 2 + 0.5) for ang in range(0, 360, 10)]
    # gradient = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    gradient = [(1, 0), (-1, 0), (0, -1), (0, -1)]
    ang = np.cos(np.radians(45))
    gradient = [(ang, -ang), (ang, ang), (ang, -ang), (ang, ang)]
    
    debugImage('inPoly', inPoly(x, y, points, gradient), show=True)


def test():
    p1 = (0.5, 0)
    p2 = (0, 0.5)
    
    p1x, p1y = p1[0], p1[1]
    p2x, p2y = p2[0], p2[1]
    
    dPx, dPy = p2x - p1x, p2y - p1y
    
    px = (x + p1x + (dPx / dPy) * (y - p1y)) / 2
    py = ((dPy / dPx) * (x - p1x) + y + p1y) / 2
    
    xP1x, yP1y = px - p1x, py - p1y
    xP2x, yP2y = px - p2x, py - p2y
    xPx, yPy = x - px, y - py
    
    onLine = np.isclose(np.sqrt(dPx * dPx + dPy * dPy),
                        np.sqrt(xP1x * xP1x + yP1y * yP1y) + np.sqrt(
                            xP2x * xP2x + yP2y * yP2y))
    rightSide = ((dPx * yPy - dPy * xPx) >= 0) * onLine
    
    debugImage('onLine', onLine * 1, show=False)
    debugImage('px', px * onLine, show=True)
    debugImage('py', py * onLine, show=True)
    
    max1 = np.zeros(shape=(size, size))
    
    min1 = np.zeros(shape=(size, size)) + 1
    max2 = np.zeros(shape=(size, size))
    min2 = np.zeros(shape=(size, size)) + 1
    
    points = ((0, 0), (0.75, 0), (0, 1), (0.75, 1))
    maxes = [max(np.math.sqrt(
        (p[0] - p2[0]) * (p[0] - p2[0]) + (p[1] - p2[1]) * (p[1] - p2[1])) for
                 p2 in points) for p in points]
    for seed in range(1):
        step = 30
        p = np.zeros(shape=(2, 2), dtype=int)
        for j in range(0, 2):
            for i in range(0, 2):
                h = seed
                for c in '{},{}'.format(i, j):
                    h = (31 * h + 17 * ord(c)) & 0xFFFFFFFF
                np.random.seed(h)
                p[j][i] = np.random.randint(0, 360 // step)
        angles = np.array(
            [[np.cos(np.radians(ang)), np.sin(np.radians(ang))] for ang in
             range(0, 360, step)])
        
        grads = angles[p]
        
        const = 1 / np.sqrt(2)
        lerp = lambda a, b, x: a + x * (b - a)
        fade = lambda t: t * t * t * (t * (t * 6 - 15) + 10)
        
        grad = grads[0][0]
        workX = x - points[0][0]
        workY = y - points[0][1]
        dist1 = np.sqrt(workX * workX + workY * workY)
        dist1 = fade(1 - dist1 / maxes[0])
        workX = np.abs(workX)
        workX = workX / workX.max()
        workY = np.abs(workY)
        workY = workY / workY.max()
        dist1 = fade(1 - np.abs(workX)) * fade(1 - np.abs(workY))
        dot1 = grad[0] * workX + grad[1] * workY
        
        grad = grads[0][1]
        workX = x - points[1][0]
        workY = y - points[1][1]
        dist2 = np.sqrt(workX * workX + workY * workY)
        dist2 = fade(1 - dist2 / maxes[1])
        workX = np.abs(workX)
        workX = workX / workX.max()
        workY = np.abs(workY)
        workY = workY / workY.max()
        dist2 = fade(1 - np.abs(workX)) * fade(1 - np.abs(workY))
        dot2 = grad[0] * workX + grad[1] * workY
        
        grad = grads[1][0]
        workX = x - points[2][0]
        workY = y - points[2][1]
        dist3 = np.sqrt(workX * workX + workY * workY)
        dist3 = fade(1 - dist3 / maxes[2])
        workX = np.abs(workX)
        workX = workX / workX.max()
        workY = np.abs(workY)
        workY = workY / workY.max()
        dist3 = fade(1 - np.abs(workX)) * fade(1 - np.abs(workY))
        dot3 = grad[0] * workX + grad[1] * workY
        
        grad = grads[1][1]
        workX = x - points[3][0]
        workY = y - points[3][1]
        dist4 = np.sqrt(workX * workX + workY * workY)
        dist4 = fade(1 - dist4 / maxes[3])
        workX = np.abs(workX)
        workX = workX / workX.max()
        workY = np.abs(workY)
        workY = workY / workY.max()
        dist4 = fade(1 - np.abs(workX)) * fade(1 - np.abs(workY))
        print(workX)
        print(dist4)
        dot4 = grad[0] * workX + grad[1] * workY
        
        u = fade(x)
        v = fade(y)
        x1 = lerp(dot1, dot2, u)
        x2 = lerp(dot3, dot4, u)
        result1 = lerp(x1, x2, v)
        
        u1 = 1 - u
        v1 = 1 - v
        
        u1v1 = u1 * v1
        uv1 = u * v1
        u1v = u1 * v
        uv = u * v
        print(dist1)
        print(u1v1)
        print(dist2)
        print(uv1)
        print(dist3)
        print(u1v)
        print(dist4)
        print(uv)
        result1 = dot1 * u1v1 + dot2 * uv1 + dot3 * u1v + dot4 * uv
        
        result2 = (dist1 * dot1 + dist2 * dot2 + dist3 * dot3 + dist4 * dot4)
        
        min1 = np.minimum(result1, min1)
        max1 = np.maximum(result1, max1)
        min2 = np.minimum(result2, min2)
        max2 = np.maximum(result2, max2)
    
    mask = np.logical_and(x < pX, y < pY)
    
    debugImage('result1', result1, show=False)
    debugImage('result2', result2, show=False)
    
    debugImage('min1', min1, show=False)
    debugImage('min2', min2, show=False)
    debugImage('max1', max1, show=False)
    debugImage('max2', max2, show=False)
    
    array = np.array([[1, 1], [3, 4]], dtype=float)
    print(array)
    grad0, grad1 = np.gradient(array)
    print(grad0)
    print(grad1)
    grad0, grad1 = np.gradient(array, 0.75, 0.25)
    print(grad0)
    print(grad1)


def normalTest():
    def getNeighborsArray(array, x, y):
        neighbors = []
        for i, j in [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1),
                     (-1, 1), (-1, 0)]:
            xPos = x + i
            yPos = y + j
            
            xDir = x.copy() * 0 + i
            yDir = y.copy() * 0 + j
            
            np.add(x, 0, out=xPos, where=(xPos < 0) + (x.shape[0] <= xPos))
            np.add(y, 0, out=yPos, where=(yPos < 0) + (y.shape[1] <= yPos))
            
            print(xPos)
            print(yPos)
            
            neighbors.append(np.array([xDir / x.shape[0], yDir / y.shape[1],
                                       array[yPos, xPos] - array]))
        return np.moveaxis(np.array(neighbors), 1, 3)
    
    def colorize(color, sections=256):
        if len(color) == 1:
            color = [color[0]] * 3
        sections = 256 / sections
        for i, c in enumerate(color):
            if type(c) in [int, np.int8, np.int16, np.int32, np.int64]:
                color[i] = np.uint32(c) & 0xFF
            elif type(c) in [float, np.float16, np.float32, np.float64]:
                color[i] = np.uint32(c * 255) & 0xFF
            elif type(c) == np.ndarray:
                if c.dtype in [np.int8, np.int16, np.int32, np.int64, np.uint8,
                               np.uint16, np.uint32, np.uint64]:
                    color[i] = c.astype('uint32') & 0xFF
                elif c.dtype in [np.float16, np.float32, np.float64]:
                    color[i] = (c * 255).astype('uint32') & 0xFF
            color[i] = (color[i] // sections * sections).astype('uint32')
        while len(color) < 4:
            color.append(np.uint32(255))
        return (color[0] << 0) + (color[1] << 8) + (color[2] << 16) + (
                    color[3] << 24)
    
    size = 128
    
    lin = np.linspace(0, 1, size, endpoint=False)
    x, y = np.meshgrid(lin, lin)
    
    xInt, yInt = (x * size).astype(int), (y * size).astype(int)
    
    fade = lambda t: t * t * t * (t * (t * 6 - 15) + 10)
    
    # array = fade(x) ** 1 * fade(y) ** 1
    r = 0.5
    array = np.sqrt(r * r - (x - 0.5) * (x - 0.5) - (y - 0.5) * (y - 0.5)) * 5
    # array = r - np.sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5))
    # array = np.where(array >= 0, array, 0) * 10
    
    loadImg = Image.open('03 - WL.png')
    array = np.asarray(loadImg).T[0] / 255
    
    img = Image.fromarray(colorize([array]), 'RGBA')
    img.save('test.png', 'PNG')
    
    r = array.copy() * 0
    g = array.copy() * 0
    b = array.copy() * 0
    
    neighbors = getNeighborsArray(array, xInt, yInt)
    for i, p2 in enumerate(neighbors):
        p1 = neighbors[i - 1]
        
        cross = np.cross(p1[yInt, xInt], p2[yInt, xInt])
        cross = (cross / np.linalg.norm(cross, axis=2).reshape(
            (size, size, 1)) + 1) / 2
        
        r += cross[:, :, 0]
        g += cross[:, :, 1]
        b += (cross[:, :, 2] + 1) / 2
    
    r, g, b = r / 8, g / 8, b / 8
    
    print(array)
    print(r)
    print(g)
    print(b)
    
    img = Image.fromarray(colorize([r, g, b]), 'RGBA')
    img.save('rgb.png', 'PNG')


if __name__ == '__main__':
    perlinNoise = ns.PerlinArray(1, seed=10, octaves=2, persistence=0.9)
    
    # Common
    size = 128
    radius = 10
    seed = 3
    
    
    def worldFunc(x, y, array):
        return np.abs(2 * array - 1)
    
    
    # def worldFunc(x, y, array):
    # array *= 4
    # return array - array.astype(int)
    # def worldFunc(x, y, array):
    # lin = np.linspace(0, 1, array.shape[0], endpoint = False)
    # xVal, yVal = np.meshgrid(lin, lin)
    # xVal += x
    # yVal = perlinNoise.calculateValues((yVal + y + 10) / 10)
    # return ((np.sin(xVal / 2 + np.math.pi * np.cos(yVal / 3) / 1.5) + 1) / 2 + array) / 2
    # return ((np.sin(xVal / 1.5 + yVal * 8) + 1) / 2 + array) / 2
    # worldFunc = None
    octaves = 6
    persistence = 0.75
    sections = 256
    
    # Diamond Square
    period = 2 ** 3
    
    
    def dsFunc(array, size, x, y, row, col, r, h, p, rand):
        xVal = x * size + col - size / 2
        yVal = y * size + row - size / 2
        dist = np.sqrt(
            xVal * xVal + yVal * yVal) if xVal != 0 or yVal != 0 else 0.0000001
        val = h + (rand - 0.5) * r / p * size / dist
        return min(max(val, 0.0), 1.0)
    
    
    dsFunc = None
    
    # Perlin
    step = 45
    
    # Worley
    count = 1
    correction = 1.0
    
    
    def wlFunc(tile, array):
        return array[:, :, 1] - array[:, :, 0]
    
    
    def wlFunc(tile, array):
        return array[:, :, 1] * array[:, :, 0]
    
    
    wlFunc = None
    
    # world = ns.World('01 - DS', size = size, tile = ns.DSTile, seed = seed, worldFunc = worldFunc, period = period, func = dsFunc)
    # world.create(radius)
    # world.toImage(sections = sections)
    
    # del world
    
    # world = ns.World('02 - PL', size = size, tile = ns.PerlinTile, seed = seed, worldFunc = worldFunc, octaves = octaves, persistence = persistence, step = step)
    # world.create(radius)
    # world.toImage(sections = sections)
    
    # del world
    
    # world = ns.World('03 - WL', size = size, tile = ns.WorleyTile, seed = seed, worldFunc = worldFunc, octaves = octaves, persistence = persistence, count = count, correction = correction, func = wlFunc)
    # world.create(radius)
    # world.toImage(sections = sections)
    
    # del world
    
    # world = ns.World('04 - VL', size = size, tile = ns.ValueTile, seed = seed, worldFunc = worldFunc, octaves = octaves, persistence = persistence)
    # world.create(radius)
    # world.toImage(sections = sections)
    
    # del world
    
    # world = ns.World('05 - OS', size = size, tile = ns.OpenSimplexTile, seed = seed, worldFunc = worldFunc, octaves = octaves, persistence = persistence)
    # world.create(radius)
    # world.toImage(sections = sections)
    
    # del world
    
    # inPolyTest()
    # test()
    normalTest()
