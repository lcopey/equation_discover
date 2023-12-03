from equation_discover import BASE_TOKENS, EquationSampler

if __name__ == "__main__":
    n_samples = 32
    sampler = EquationSampler(BASE_TOKENS, 16, 2)
    sequences, entropies, log_probs, counters, lengths = sampler.sample_sequence(
        n_samples, repeat=1
    )
    sampler.parent_sibling(sequences=sequences, lengths=lengths)
