#%%
from foo import *

#%%
r = []
for epsilon in [ 0.1, 0.5, 1, 2, 4]:
    results = []
    for i in range(100):
        bar = UCB(epsilon)
        results.append(bar.simulate(1000))

    results = np.array(results)
    results = np.mean(results, axis=0)
    r.append(results)

for f, l in zip(r, [0, 0.01, 0.1]):
    plt.plot(f, label=l)
plt.legend()
plt.show()
# %%
