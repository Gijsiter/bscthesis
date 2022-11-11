from os import walk
import re
from random import sample, shuffle
import numpy as np
import pickle as pk


names = []
for (_, _, filenames) in walk('audio/'):
    names.extend(filenames)
    break

set_names = ['train', 'val']
sets = dict()

for set_name in set_names:
    samps = []
    for fam in set([s.split('_')[0] for s in names]):

        r = re.compile(f"{fam}.*")
        bruh = list(filter(r.match, names))
        srcs = set([s.split('_')[1] for s in bruh])

        if len(srcs) == 1:
            ams = [15]
        elif len(srcs) == 2:
            ams = [8, 7]
        else:
            ams = [5, 5, 5]

        for src, am in zip(srcs, ams):
            r = re.compile(f".*{src}.*")
            sel = list(filter(r.match, bruh))
            sorted_notes = sorted(sel, key=lambda s: s.split('-')[1])
            for subset in np.array_split(sorted_notes, am):
                samps.append(sample(list(subset), 1)[0])

    shuffle(samps)
    sets[set_name] = samps
    names = [name for name in names if name not in samps]

pk.dump(sets, open('datasets.pk', 'wb'))
