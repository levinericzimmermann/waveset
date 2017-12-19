import soundfile as sf
import numpy as np
import itertools
import random


class Waveset(list):
    @property
    def max_jump(self):
        if len(self) > 1:
            return max(abs(w1 - w2) for w1, w2 in zip(self, self[1:]))
        else:
            return 0

    @property
    def absmax(x):
        return max(max(x), abs(min(x)))

    @property
    def summax(x):
        return sum(map(abs, x)) / len(x)

    @property
    def most(self):
        return max(set(self), key=self.count)

    def get_harmony(self, factor):
        assert(factor >= 1)
        remainder = len(self) % factor
        lens = [len(self) // factor] * factor
        lens[-1] += remainder
        assert(sum(lens) == len(self))
        result = []
        for l in lens:
            result = result + list(self.stretch(l))
        return result

    def harmonize(self, *factor):
        n = len(factor)
        harms = map(lambda f: self.get_harmony(f), factor)
        return [sum(x) / n for x in zip(*harms)]

    def stretch(self, length):
        x = np.linspace(0, len(self) - 1, length)
        xp = [i for i in range(len(self))]
        return type(self)(np.interp(x, xp, self))

    def similar(self, ws):
        result = np.correlate(self, ws)
        return max(result)

    def interpolate(self, ws, weight_x):
        assert(len(self) == len(ws))
        weight_y = 1 - weight_x
        return [weight_x * x + weight_y * y for x, y in zip(self, ws)]

    def scale(self, fac):
        return type(self)(map(lambda x: x * fac, self))

    def reverse(self):
        return type(self)(reversed(self))


class Wavesets(list):
    def __init__(self, iterable, first, last, sr):
        self.sr = sr
        self.first = first
        self.last = last
        list.__init__(self, iterable)

    def __getitem__(self, idx):
        return self.halfcopy(list.__getitem__(self, idx))

    @staticmethod
    def wav_to_list(name):
        with sf.SoundFile(name, 'r') as f:
            assert(f._info.channels == 1)
            sr = f._info.samplerate
            data = list(f.read(len(f)))
        return data, sr

    @staticmethod
    def is_cross(x, y, cross):
        posneg = (x >= cross) and (cross > y)
        negpos = (y >= cross) and (cross > x)
        return posneg or negpos

    @staticmethod
    def get_wavesets(wave, cross, ncrossings):
        result = []
        firstover = False
        first = []
        last = []
        waveset = []
        counter = 0
        for w1, w2 in zip(wave, wave[1:]):
            waveset.append(w1)
            if Wavesets.is_cross(w1, w2, cross):
                if counter == ncrossings or not(firstover):
                    if firstover:
                        result.append(Waveset(waveset))
                    else:
                        first = Waveset(waveset)
                        firstover = True
                    waveset = []
                    counter = 0
                counter += 1
        last = Waveset(waveset + [wave[-1]])
        return first, result, last

    @staticmethod
    def linspace(start, end, n):
        return np.linspace(start, end, n + 2)[1:-1]

    @staticmethod
    def lens_n(len1, len2, n):
        return Wavesets.linspace(len1, len2, n)

    @staticmethod
    def lens_step(len1, len2, step):
        n = abs(len1 - (len2 - step)) // step
        return Wavesets.lens_n(len1, len2, n)

    @staticmethod
    def lens_rel(len1, len2, rel):
        step = len1 / rel
        return Wavesets.lens_step(len1, len2, step)

    @staticmethod
    def lens_quant(len1, len2, lens_func, quants):
        quants1 = list(map(lambda x: len1 * x, quants))
        quants2 = list(map(lambda x: len2 * x, quants))
        quants = quants1 + quants2
        quants = list(filter(lambda x: (len1 <= x and x <= len2)
                             or len2 <= x and x <= len1, quants))
        lens = lens_func(len1, len2)
        result = list(map(lambda x: Wavesets.quantize(x, quants), lens))
        return result

    @staticmethod
    def quantize(x, quants):
        return min(quants, key=lambda y: abs(y - x))

    @classmethod
    def from_wav(cls, name, cross, ncrossings):
        wave, sr = Wavesets.wav_to_list(name)
        first, wavesets, last = Wavesets.get_wavesets(wave, cross, ncrossings)
        return cls(wavesets, first, last, sr)

    @property
    def length(self):
        return sum(len(ws) for ws in self) + len(
                self.first) + len(self.last)

    @property
    def duration(self):
        return self.length / self.sr

    def cut_last(self):
        return type(self)(self[:-1], self.first, self[-1], self.sr)

    def to_wav(self, name, norm=1):
        wave = [i for sub in self for i in sub]
        wave = self.first + wave + self.last
        wave = list(map(lambda x: x * norm, wave))
        with sf.SoundFile(name, 'x', self.sr, 1, 'PCM_24') as f:
            f.write(wave)
        return name

    def halfcopy(self, wavesets):
        return type(self)(wavesets, self.first, self.last, self.sr)

    def copy(self):
        return self.halfcopy(self[:])

    def shuffle(self):
        result = []
        firsts = itertools.islice(self, 0, None, 2)
        seconds = itertools.islice(self, 1, None, 2)
        for w1, w2 in zip(firsts, seconds):
            result.append(w2)
            result.append(w1)
        return self.halfcopy(result)

    def sort_by_maxjump(self):
        return self.halfcopy(sorted(self, key=lambda ws: ws.max_jump))

    def sort_by_similarity(self):
        ws = self.copy()
        last = ws[0]
        ws.remove(last)
        result = [last]
        while len(ws) > 0:
            last = max(ws, key=lambda w: w.similar(last))
            ws.remove(last)
            result.append(last)
        return self.halfcopy(result)

    def sort_by_maxavg(self):
        def maxavg(x): return (x.summax + x.absmax) * 0.5
        return self.halfcopy(sorted(self, key=maxavg))

    def sort_by_most(self):
        return self.halfcopy(sorted(self, key=lambda ws: ws.most))

    def sort_by_max(self):
        return self.halfcopy(sorted(self, key=lambda ws: ws.absmax))

    def sort_by_avg(self):
        return self.halfcopy(sorted(self, key=lambda ws: ws.summax))

    def harmonize(self, *factor):
        return self.halfcopy(map(lambda ws: ws.harmonize(*factor), self))

    def smooth(self, lens_func):
        def stretches(ws, lens):
            return [ws.stretch(l) for l in lens]
        result = []
        for w1, w2 in zip(self, self[1:]):
            lens = lens_func(len(w1), len(w2))
            stretches1 = stretches(w1, lens)
            stretches2 = list(reversed(stretches(w2, reversed(lens))))
            weights = type(self).linspace(1, 0, len(stretches1))
            trans = [w1]
            for s1, s2, w in zip(stretches1, stretches2, weights):
                t = s1.interpolate(s2, w)
                trans.append(t)
            result = result + trans
        result.append(self[-1])
        return self.halfcopy(result)

    def stretch(self, fac):
        return self.halfcopy(ws.stretch(fac) for ws in self[:])

    def repeat(self, am, scale_range=0):
        result = []
        for ws in self[:]:
            for i in range(am):
                scaled = ws.scale(random.uniform(1-scale_range, 1))
                result.append(scaled)
        return self.halfcopy(result)

    def dynamic_repeat(self, start_repeat, end_repeat, scale_range=0):
        repeats = map(int, Wavesets.linspace(start_repeat, end_repeat,
                                             len(self)))
        result = []
        for ws, am in zip(self[:], repeats):
            for i in range(am):
                scaled = ws.scale(random.uniform(1-scale_range, 1))
                result.append(scaled)
        return self.halfcopy(result)

    def reversed_sets(self):
        return self.halfcopy(ws.reverse() for ws in self[:])

    def distort(self, fac):
        return self.halfcopy(ws.scale(fac) for ws in self[:])

    def interleave(self, other):
        assert(self.sr == other.sr)
        result = []
        for ws0, ws1 in zip(self[:], other[:]):
            result.append(ws0)
            result.append(ws1)
        return self.halfcopy(result)

    def scale_randomly(self, range):
        return self.halfcopy(ws.scale(random.uniform(1-range, 1))
                             for ws in self[:])

    def dynamic_scale(self, startscale, endscale):
        factors = Wavesets.linspace(
                startscale, endscale, len(self))
        result = []
        for ws, fac in zip(self[:], factors):
            result.append(ws.scale(fac))
        return self.halfcopy(result)
