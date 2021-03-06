# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from fairlearn.reductions import GridSearch, DemographicParity

import copy
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _simple_threshold_data(number_a0, number_a1,
                           a0_threshold, a1_threshold,
                           a0_label, a1_label):

    a0s = np.full(number_a0, a0_label)
    a1s = np.full(number_a1, a1_label)

    a0_scores = np.linspace(0, 1, number_a0)
    a1_scores = np.linspace(0, 1, number_a1)
    score_feature = np.concatenate((a0_scores, a1_scores), axis=None)

    A = np.concatenate((a0s, a1s), axis=None)

    Y_a0 = [x > a0_threshold for x in a0_scores]
    Y_a1 = [x > a1_threshold for x in a1_scores]

    Y = np.concatenate((Y_a0, Y_a1), axis=None)

    X = pd.DataFrame({"actual_feature": score_feature,
                      "sensitive_features": A,
                      "constant_ones_feature": np.ones(len(Y))})
    return X, Y, A


def test_demographicparity_fair_uneven_populations():
    # Variant of test_demographicparity_already_fair, which has unequal
    # populations in the two classes
    # Also allow the threshold to be adjustable

    score_threshold = 0.625

    number_a0 = 4
    number_a1 = 4

    a0_label = 17
    a1_label = 37

    X, Y, A = _simple_threshold_data(number_a0, number_a1,
                                     score_threshold, score_threshold,
                                     a0_label, a1_label)

    target = GridSearch(LogisticRegression(solver='liblinear', fit_intercept=True),
                        constraints=DemographicParity(),
                        grid_size=11)

    target.fit(X, Y, sensitive_features=A)
    assert len(target.all_results) == 11

    test_X = pd.DataFrame({"actual_feature": [0.2, 0.7],
                           "sensitive_features": [a0_label, a1_label],
                           "constant_ones_feature": [1, 1]})

    sample_results = target.predict(test_X)
    sample_proba = target.predict_proba(test_X)
    assert np.allclose(sample_proba, [[0.53748641, 0.46251359], [0.46688736, 0.53311264]])

    sample_results = target.all_results[0].predictor.predict(test_X)
    assert np.array_equal(sample_results, [1, 0])


def test_lambda_vec_zero_unchanged_model():
    score_threshold = 0.6

    number_a0 = 64
    number_a1 = 24

    a0_label = 7
    a1_label = 22

    X, y, A = _simple_threshold_data(number_a0, number_a1,
                                     score_threshold, score_threshold,
                                     a0_label, a1_label)

    estimator = LogisticRegression(solver='liblinear',
                                   fit_intercept=True,
                                   random_state=97)

    # Train an unmitigated estimator
    unmitigated_estimator = copy.deepcopy(estimator)
    unmitigated_estimator.fit(X, y)

    # Do the grid search with a zero Lagrange multiplier
    iterables = [['+', '-'], ['all'], [a0_label, a1_label]]
    midx = pd.MultiIndex.from_product(iterables, names=['sign', 'event', 'group_id'])
    lagrange_zero_series = pd.Series(np.zeros(4), index=midx)
    grid_df = pd.DataFrame(lagrange_zero_series)

    target = GridSearch(estimator,
                        constraints=DemographicParity(),
                        grid=grid_df)
    target.fit(X, y, sensitive_features=A)
    assert len(target.all_results) == 1

    # Check coefficients
    gs_coeff = target.best_result.predictor.coef_
    um_coeff = unmitigated_estimator.coef_
    assert np.array_equal(gs_coeff, um_coeff)


def test_can_specify_and_generate_lambda_vecs():
    score_threshold = 0.4

    number_a0 = 32
    number_a1 = 24

    a0_label = 11
    a1_label = 3

    X, y, A = _simple_threshold_data(number_a0, number_a1,
                                     score_threshold, score_threshold,
                                     a0_label, a1_label)

    estimator = LogisticRegression(solver='liblinear',
                                   fit_intercept=True,
                                   random_state=97)

    iterables = [['+', '-'], ['all'], sorted([a0_label, a1_label])]
    midx = pd.MultiIndex.from_product(iterables, names=['sign', 'event', 'group_id'])
    lagrange_negative_series = pd.Series([0.0, 0.0, 0.0, 2.0], index=midx)
    lagrange_zero_series = pd.Series(np.zeros(4), index=midx)
    lagrange_positive_series = pd.Series([0.0, 2.0, 0.0, 0.0], index=midx)
    grid_df = pd.concat([lagrange_negative_series,
                         lagrange_zero_series,
                         lagrange_positive_series],
                        axis=1)

    target1 = GridSearch(copy.deepcopy(estimator),
                         constraints=DemographicParity(),
                         grid_size=3)

    target2 = GridSearch(copy.deepcopy(estimator),
                         constraints=DemographicParity(),
                         grid=grid_df)

    # Try both ways of specifying the Lagrange multipliers
    target2.fit(X, y, sensitive_features=A)
    target1.fit(X, y, sensitive_features=A)

    assert len(target1.all_results) == 3
    assert len(target2.all_results) == 3

    # Check we generated the same multipliers
    for i in range(3):
        lm1 = target1.all_results[i].lambda_vec
        lm2 = target2.all_results[i].lambda_vec
        assert lm1.equals(lm2)

    # Check the models are the same
    for i in range(3):
        coef1 = target1.all_results[i].predictor.coef_
        coef2 = target2.all_results[i].predictor.coef_
        assert np.array_equal(coef1, coef2)
