
def round_channels(n_chan, multiplier):
    new_chan = n_chan * multiplier
    new_chan = max(8, int(new_chan + 4) // 8 * 8)
    if new_chan < 0.9 * n_chan: new_chan += 8
    return new_chan


def round_repeats():
    pass
