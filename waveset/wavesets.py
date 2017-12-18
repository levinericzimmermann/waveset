import soundfile as sf


def is_cross(x, y, cross):
    posneg = (x >= cross) and (cross > y)
    negpos = (y >= cross) and (cross > x)
    return posneg or negpos


def get_wavesets(wave, cross,  ncrossings):
    result = []
    firstover = False
    first = []
    last = []
    waveset = []
    counter = 0
    for w1, w2 in zip(wave, wave[1:]):
        waveset.append(w1)
        if is_cross(w1, w2, cross):
            if counter == ncrossings or not(firstover):
                if firstover:
                    result.append(waveset)
                else:
                    first = waveset
                    firstover = True
                waveset = []
                counter = 0
            counter += 1
    last = waveset + [wave[-1]]
    return first, result, last


def wavesets_to_wav(wavesets, sr, name, norm):
    wave = [i for sub in wavesets for i in sub]
    wave = list(map(lambda x: x * norm, wave))
    with sf.SoundFile(name, 'x', sr, 1, 'PCM_24') as f:
        f.write(wave)


def wav_to_list(name):
    with sf.SoundFile(name, 'r') as f:
        assert(f._info.channels == 1)
        sr = f._info.samplerate
        data = list(f.read(len(f)))
    return data, sr


def process_file(inputname, outputname, cross, ncrossings, func):
    data, sr = wav_to_list(inputname)
    norm = max(data)
    data = list(map(lambda x: x / norm, data))
    first, wavesets, last = get_wavesets(data, cross, ncrossings)
    wavesets = func(wavesets)
    wavesets_to_wav([first] + wavesets + [last], sr, outputname, 1)
