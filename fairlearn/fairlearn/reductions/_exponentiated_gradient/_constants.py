# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

_PRECISION = 1e-8
_LINE = "_" * 9
_INDENTATION = " " * 9


# Explicit optimization parameters of expgrad

# A multiplier controlling the automatic setting of nu.
_ACCURACY_MUL = 0.5

# Parameters controlling adaptive shrinking of the learning rate.
_REGRET_CHECK_START_T = 5
_REGRET_CHECK_INCREASE_T = 1.6
_SHRINK_REGRET = 0.8
_SHRINK_ETA = 0.8

# The smallest number of iterations after which expgrad terminates.
_MIN_T = 5

# If _RUN_LP_STEP is set to True, then each step of exponentiated
# gradient is followed by the saddle point optimization over the convex
# hull of classifiers returned so far.
_RUN_LP_STEP = True
