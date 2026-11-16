"""Behavior label sets, and the two scoring protocols that compare models across label sets."""

BEHAVIORS: tuple[str, ...] = ("resting", "walking", "rearing", "grooming", "eating")

# Real datasets add an "other" class for labelled activity outside the core set
# (nesting, digging, object play, ...).
BEHAVIORS_EXT: tuple[str, ...] = BEHAVIORS + ("other",)

# The classes every label set and every classifier agrees on, and the set real data labels:
# the two protocols behavior models are scored on (a synthetic-only model has no "other").
SHARED: tuple[str, ...] = ("walking", "rearing", "grooming", "eating")
LABELLED: tuple[str, ...] = SHARED + ("other",)
