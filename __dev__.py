from equation_discover import BASE_TOKENS, RNNSampler

if __name__ == "__main__":
    n_samples = 32
    sampler = RNNSampler(BASE_TOKENS, 16, 2)
    sequences, entropies, log_probs, counters, lengths = sampler.sample(
        n_samples, repeat=1
    )
    sampler.parent_sibling(sequences=sequences, lengths=lengths)
