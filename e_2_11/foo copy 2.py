#%%
from foo import *

#%%
n = 10
stationary = True

self = GradientBandit(0.0625)
self.q = np.random.normal(0, 1, k)
rewards = np.zeros(n)
for t in range(1, n +1):
    if not stationary:
        self.q += np.random.normal(0, 0.01, k)
    t0 = time.time()
    A = self.select_action(t)
    print(f"sel: {round(time.time()-t0, 6)}")

    R = np.random.normal(self.q[A])
    self.N[A] += 1

    t0 = time.time()
    self.update(A, R, t)
    print(f"up: {round(time.time()-t0, 6)}")

    rewards[t-1] = R

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
