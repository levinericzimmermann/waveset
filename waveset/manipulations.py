import numpy as np
from itertools import islice


# manipulations


def jump_wavesets(ws):
    return sorted(ws, key=max_jump)


def similiar_wavesets(ws):
    last = ws[0]
    ws.remove(last)
    result = [last]
    while len(ws) > 0:
        last = max(ws, key=lambda w: similiar(last, w))
        ws.remove(last)
        result.append(last)
    return result


def harmonize_wavesets(xs, factors):
    return list(map(lambda x: harmonize(x, factors), xs))


def shuffle_wavesets(xs):
    result = []
    firsts = islice(xs, 0, None, 2)
    seconds = islice(xs, 1, None, 2)
    for w1, w2 in zip(firsts, seconds):
        result.append(w2)
        result.append(w1)
    return result


def maxavg_wavesets(xs):
    def maxavg(x):
        return (summax(x) + absmax(x)) * 0.5
    return sorted(xs, key=maxavg)


def most_wavesets(xs):
    return sorted(xs, key=most)


def max_wavesets(xs):
    return sorted(xs, key=absmax)


def avg_wavesets(xs):
    return sorted(xs, key=summax)


def smooth_wavesets(wavesets, lens_func):
    result = []
    for w1, w2 in zip(wavesets, wavesets[1:]):
        lens = lens_func(len(w1), len(w2))
        stretches1 = stretches(w1, lens)
        stretches2 = list(reversed(stretches(w2, reversed(lens))))
        weights = my_linspace(1, 0, len(stretches1))
        trans = [w1]
        for s1, s2, w in zip(stretches1, stretches2, weights):
            t = interpolate(s1, s2, w)
            trans.append(t)
        result = result + trans
    result.append(wavesets[-1])
    return result


# lens

def lens_n(len1, len2, n):
    return my_linspace(len1, len2, n)


def lens_step(len1, len2, step):
    n = abs(len1 - (len2 - step)) // step
    return lens_n(len1, len2, n)


def lens_rel(len1, len2, rel):
    step = len1 / rel
    return lens_step(len1, len2, step)


def lens_quant(len1, len2, lens_func, quants):
    quants1 = list(map(lambda x: len1 * x, quants))
    quants2 = list(map(lambda x: len2 * x, quants))
    quants = quants1 + quants2
    quants = list(filter(lambda x: (len1 <= x and x <= len2)
                         or len2 <= x and x <= len1, quants))
    lens = lens_func(len1, len2)
    result = list(map(lambda x: quantize(x, quants), lens))
    return result


# utility

def quantize(x, quants):
    return min(quants, key=lambda y: abs(y - x))


def my_linspace(start, end, n):
    return np.linspace(start, end, n + 2)[1:-1]


def most(xs):
    return max(set(xs), key=xs.count)


def interpolate(xs, ys, weight_x):
    assert(len(xs) == len(ys))
    weight_y = 1 - weight_x
    return [weight_x * x + weight_y * y for x, y in zip(xs, ys)]


def stretch(xs, length):
    x = np.linspace(0, len(xs) - 1, length)
    xp = [i for i in range(len(xs))]
    return np.interp(x, xp, xs)


def stretches(xs, lens):
    return [stretch(xs, l) for l in lens]


def get_harmony(xs, factor):
    assert(factor >= 1)
    remainder = len(xs) % factor
    lens = [len(xs) // factor] * factor
    lens[-1] += remainder
    assert(sum(lens) == len(xs))
    result = []
    for l in lens:
        result = result + list(stretch(xs, l))
    return result


def harmonize(xs, factors):
    n = len(factors)
    harms = map(lambda f: get_harmony(xs, f), factors)
    return [sum(x) / n for x in zip(*harms)]


def absmax(x):
    return max(max(x), abs(min(x)))


def summax(x):
    return sum(map(abs, x)) / len(x)


def similiar(x, y):
    result = np.correlate(x, y)
    return max(result)


def max_jump(ws):
    jumps = []
    for w1, w2 in zip(ws, ws[1:]):
        jumps.append(abs(w1 - w2))
    return max(jumps)
