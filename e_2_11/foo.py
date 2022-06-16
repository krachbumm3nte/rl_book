#%%
import numpy as np
import matplotlib.pyplot as plt
import time
k = 10


# %%


class BanditAlgorithm:

    def __init__(self) -> None:
        self.N = np.zeros(k)
        self.Q = np.zeros(k)


    def select_action(self, t):
        pass

    def update(self, A, R):
        pass

    def simulate(self, n, stationary=True):
        self.q = np.random.normal(0, 1, k)
        rewards = np.zeros(n)
        for t in range(1, n +1):
            if not stationary:
                self.q += np.random.normal(0, 0.01, k)
            A = self.select_action(t)
            
            R = np.random.normal(self.q[A])
            self.N[A] += 1

            self.update(A, R, t)
            rewards[t-1] = R
        return rewards

class EpsilonGreedy(BanditAlgorithm):

    def __init__(self, epsilon) -> None:
        super().__init__()
        self.epsilon = epsilon


    def select_action(self, t):
        if np.random.random() > self.epsilon:
            # exploit
            return np.argmax(self.Q)
        else:
            # explore
            return np.random.randint(k)
        
    def update(self, A, R, t):
        self.Q[A] = self.Q[A] + (1/self.N[A]) * (R - self.Q[A])


class OptimisticGreedy(EpsilonGreedy):

    def __init__(self, Q_1) -> None:
        super().__init__(0.)
        self.Q = np.ones(k) * Q_1


class UCB(BanditAlgorithm):

    def __init__(self, c) -> None:
        super().__init__()
        self.c = c

    def select_action(self, t):
        foo = self.q + self.c * np.sqrt(np.log(t)/self.N)
        # TODO: handle zero div runtime warning
        foo[np.isnan(foo)] = np.inf
        return np.argmax(foo)

    def update(self, A, R, t):
        self.Q[A] = self.Q[A] + (1/self.N[A]) * (R - self.Q[A])


class GradientBandit(BanditAlgorithm):

    def __init__(self, alpha) -> None:
        super().__init__()
        self.H = np.zeros(k)
        self.alpha = alpha
        self.R_bar = 0


    def select_action(self, t):
        
        ex = np.e ** self.H

        self.Pr_A = ex / np.sum(ex)

        return np.random.choice(k, p=self.Pr_A)

    def update(self, A, R, t):
        beta = self.alpha * (R - self.R_bar)
        for a in range(k):
            if a == A:
                self.H[a] += beta * (1- self.Pr_A[a])
            else:
                self.H[a] -= beta * self.Pr_A[a]

        self.R_bar += (1/t) * (R - self.R_bar)



# %%

if __name__ == "__main__":
    configs = {
        GradientBandit: np.arange(-5, 3),
        EpsilonGreedy: np.arange(-9, -2),
        UCB: np.arange(-4, 2),
        OptimisticGreedy: np.arange(-3, 3)
    }

    for alg, params in configs.items():
        configs[alg] = 2** params.astype(float)

    all_rewards = []

    now = time.time()

    for algorithm, params in configs.items():
        print(algorithm.__name__)
        rewards = []
        for param in params:
            print(f"{np.round(time.time()-now, 2)}: parameter {param}...")
            bar = []
            for i in range(500):

                instance = algorithm(param)
                foo = instance.simulate(4000)
                bar.append(foo[:])
            bar=np.array(bar)
            bar.shape
            rewards.append(np.mean(bar))
        all_rewards.append(rewards)
        print(f"finished in {time.time()-now}ms.")
        now = time.time()

    for (alg, params), reward in zip(configs.items(), all_rewards):
        plt.plot(params, reward, label=alg.__name__)
    plt.semilogx(base=2)
    plt.legend()
    plt.show()



# %%
