import numpy as np
rng = np.random.RandomState([2015,10,10])
print rng.binomial(1, 1.0, size=(5,5))
for rnd_id in np.random.randint(low=0, high=50000, size=5):
    print rnd_id

print np.random.rand(400)