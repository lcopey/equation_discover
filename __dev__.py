import pandas as pd
import numpy as np
from equation_discover import BASE_TOKENS, RNNSampler

if __name__ == "__main__":
    n_samples = 32
    X = pd.DataFrame(np.linspace(-2 * np.pi, 2 * np.pi), columns=["var_x"])
    y = np.sin((X * 2 + 1).squeeze())

    sampler = RNNSampler(BASE_TOKENS, 16, 2)
    sampler.sample(3)
