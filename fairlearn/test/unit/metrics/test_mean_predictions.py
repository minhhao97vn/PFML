# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import fairlearn.metrics as metrics


def test_mean_prediction_unweighted():
    y_pred = [0, 1, 2, 3, 4]
    y_true = None

    result = metrics.mean_prediction(y_true, y_pred)

    assert result == 2


def test_mean_prediction_weighted():
    y_pred = [0, 1, 2, 3, 4]
    y_true = None
    weight = [8, 2, 1, 2, 1]

    result = metrics.mean_prediction(y_true, y_pred, weight)

    assert result == 1


def test_mean_overprediction_unweighted():
    y_pred = [0, 1, 2, 3, 4]
    y_true = [1, 1, 5, 0, 2]

    result = metrics.mean_overprediction(y_true, y_pred)

    assert result == 1


def test_mean_overprediction_weighted():
    y_pred = [0, 1, 2, 3, 4]
    y_true = [1, 1, 5, 0, 2]
    weight = [3, 1, 7, 1, 2]

    result = metrics.mean_overprediction(y_true, y_pred, weight)

    assert result == 0.5


def test_mean_underprediction_unweighted():
    y_pred = [0, 1, 1, 3, 4]
    y_true = [1, 1, 5, 0, 2]

    result = metrics.mean_underprediction(y_true, y_pred)

    assert result == 1


def test_mean_underprediction_weighted():
    y_pred = [0, 1, 5, 3, 1]
    y_true = [1, 1, 2, 0, 2]
    weight = [4, 1, 2, 2, 1]

    result = metrics.mean_underprediction(y_true, y_pred, weight)

    assert result == 0.5
