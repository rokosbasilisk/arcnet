# deterministic_utils.py

import itertools

# Initialize deterministic sequences for controlled variability
_choice_cycle = itertools.cycle([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
_unifint_cycle = itertools.cycle(range(1, 10))
_randint_cycle = itertools.cycle(range(0, 100))
_sample_cycle = itertools.cycle([1, 3, 5, 7, 9])

def choice(seq):
    """Pseudo-deterministic choice function based on a cycling index."""
    index = next(_choice_cycle) % len(seq) if seq else 0
    return seq[index]

def unifint(bounds):
    """Pseudo-deterministic unifint based on a cycling value."""
    a, b = bounds
    value = next(_unifint_cycle)
    return max(a, min(b, a + value))  # Fits within bounds

def randint(a, b):
    """Pseudo-deterministic randint based on a cycling value."""
    value = next(_randint_cycle)
    return a + (value % (b - a + 1))

def sample(seq, k):
    """Pseudo-deterministic sample function based on the cycling sample size."""
    size = min(k, len(seq))
    return seq[:size] if size > 0 else []

