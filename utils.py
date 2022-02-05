import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from tempeh.configurations import datasets


def load_compas(measure):
    # Load dataset and select sensitive feature
    # For Compas, sensitive feature is 'race'
    dataset = datasets['compas']()
    X_train, X_test = dataset.get_X(format=pd.DataFrame)
    y_train, y_test = dataset.get_y(format=pd.Series)
    sensitive_features_train, sensitive_features_test = dataset.get_sensitive_features('race', format=pd.Series)

    # Convert to numpy type
    x_train = X_train.to_numpy()
    x_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    g_train = sensitive_features_train.to_numpy()
    g_test = sensitive_features_test.to_numpy()
    _, g_train = np.unique(g_train, return_inverse=True)
    _, g_test = np.unique(g_test, return_inverse=True)

    # Create X, Y, G to hold entire data
    X = np.concatenate((x_train, x_test), axis=0)
    Y = np.append(y_train, y_test)
    G = np.append(g_train, g_test)

    # For Equalized odds, sensitive attribute is included in X
    if measure == 'equalized_odds':
        X = np.concatenate([X, G.reshape(-1, 1)], axis=1)

    return X, Y, G


def load_adult(measure):
    # Load dataset and select sensitive feature
    # Sensitive feature is 'sex'
    adult_train = pd.read_csv('data\\adult\\adult.data', sep=',', header=None)
    adult_test = pd.read_csv('data\\adult\\adult.test', sep='\t', header=None)
    num_features = adult_test.shape[1]
    sensitive_feature_idx = 9  # Sex feature has index = 9, Male is encoded as 1

    X_train = adult_train.drop([num_features - 1, sensitive_feature_idx], axis=1)
    X_test = adult_test.drop([num_features - 1, sensitive_feature_idx], axis=1)
    y_train = pd.Series(adult_train[num_features - 1])
    y_test = adult_test[num_features - 1]
    sensitive_features_train = adult_train[sensitive_feature_idx]
    sensitive_features_test = adult_test[sensitive_feature_idx]

    var_1_dummies = pd.get_dummies(X_train[1], prefix="workclass")
    X_train = X_train.drop(1, axis=1)
    X_train = pd.concat([X_train, var_1_dummies], axis=1)

    var_3_dummies = pd.get_dummies(X_train[3], prefix="edu")
    X_train = X_train.drop(3, axis=1)
    X_train = pd.concat([X_train, var_3_dummies], axis=1)

    var_5_dummies = pd.get_dummies(X_train[5], prefix="marital_status")
    X_train = X_train.drop(5, axis=1)
    X_train = pd.concat([X_train, var_5_dummies], axis=1)

    var_6_dummies = pd.get_dummies(X_train[6], prefix="occupation")
    X_train = X_train.drop(6, axis=1)
    X_train = pd.concat([X_train, var_6_dummies], axis=1)

    var_7_dummies = pd.get_dummies(X_train[7], prefix="relationship")
    X_train = X_train.drop(7, axis=1)
    X_train = pd.concat([X_train, var_7_dummies], axis=1)

    var_8_dummies = pd.get_dummies(X_train[8], prefix="race")
    X_train = X_train.drop(8, axis=1)
    X_train = pd.concat([X_train, var_8_dummies], axis=1)

    var_13_dummies = pd.get_dummies(X_train[13], prefix="country")
    X_train = X_train.drop(13, axis=1)
    X_train = pd.concat([X_train, var_13_dummies], axis=1)

    var_1_dummies = pd.get_dummies(X_test[1], prefix="workclass")
    X_test = X_test.drop(1, axis=1)
    X_test = pd.concat([X_test, var_1_dummies], axis=1)

    var_3_dummies = pd.get_dummies(X_test[3], prefix="edu")
    X_test = X_test.drop(3, axis=1)
    X_test = pd.concat([X_test, var_3_dummies], axis=1)

    var_5_dummies = pd.get_dummies(X_test[5], prefix="marital_status")
    X_test = X_test.drop(5, axis=1)
    X_test = pd.concat([X_test, var_5_dummies], axis=1)

    var_6_dummies = pd.get_dummies(X_test[6], prefix="occupation")
    X_test = X_test.drop(6, axis=1)
    X_test = pd.concat([X_test, var_6_dummies], axis=1)

    var_7_dummies = pd.get_dummies(X_test[7], prefix="relationship")
    X_test = X_test.drop(7, axis=1)
    X_test = pd.concat([X_test, var_7_dummies], axis=1)

    var_8_dummies = pd.get_dummies(X_test[8], prefix="race")
    X_test = X_test.drop(8, axis=1)
    X_test = pd.concat([X_test, var_8_dummies], axis=1)

    var_13_dummies = pd.get_dummies(X_test[13], prefix="country")
    X_test = X_test.drop(13, axis=1)
    X_test = pd.concat([X_test, var_13_dummies], axis=1)
    X_test['country_ temp'] = 0

    x_train = X_train.to_numpy()
    x_test = X_test.to_numpy()
    y_train = y_train.replace(' <=50K', 0.0).replace(' >50K', 1.0).to_list()
    y_train = np.array(y_train)
    y_test = y_test.replace(' <=50K.', 0.0).replace(' >50K.', 1.0).to_list()
    y_test = np.array(y_test)
    g_train = sensitive_features_train.to_numpy()
    g_test = sensitive_features_test.to_numpy()
    _, g_train = np.unique(g_train, return_inverse=True)
    _, g_test = np.unique(g_test, return_inverse=True)

    X = np.concatenate((x_train, x_test), axis=0)
    Y = np.append(y_train, y_test)
    G = np.append(g_train, g_test)

    # For Equalized odds, sensitive attribute is included in X
    if measure == 'equalized_odds':
        X = np.concatenate([X, G.reshape(-1, 1)], axis=1)
    return X, Y, G


# generate data
def generate_data(dataset, measure, random_seed):
    if dataset == 'compas':
        X, Y, G = load_compas(measure=measure)
        kernel = 'rbf'
        clean_ratio = 0.6
        attacker_split = 1.0 / 6
        test_split = 0.2
    else:
        X, Y, G = load_adult(measure=measure)
        kernel = 'rbf'
        clean_ratio = 0.9
        attacker_split = 1.0 / 2
        test_split = 0.3

    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)

    n_clean = int(clean_ratio * Y.shape[0])
    n_attacker = int(n_clean * attacker_split)

    if kernel == 'linear':
        model = LinearSVC(max_iter=10000)
    else:
        model = SVC(gamma='auto')

    model.fit(X_scaled, Y)

    loss = np.absolute(np.subtract(model.predict(X_scaled), Y))

    index = np.argsort(loss, axis=0)

    x_clean = X_scaled[index[:n_clean], :]
    y_clean = Y[index[:n_clean]]
    g_clean = G[index[:n_clean]]

    x_noise = X_scaled[index[n_clean:], :]
    y_noise = Y[index[n_clean:]]
    g_noise = G[index[n_clean:]]

    attacker_index = np.random.choice(y_clean.shape[0], n_attacker, replace=False)
    x_attacker = x_clean[attacker_index]
    y_attacker = y_clean[attacker_index]
    g_attacker = g_clean[attacker_index]
    x_clean = np.delete(x_clean, attacker_index, 0)
    y_clean = np.delete(y_clean, attacker_index, 0)
    g_clean = np.delete(g_clean, attacker_index, 0)

    x_train, x_test, y_train, y_test, g_train, g_test = train_test_split(x_clean, y_clean, g_clean,
                                                                         test_size=test_split, random_state=random_seed)

    data = {
        'x_train': x_train,
        'y_train': y_train,
        'g_train': g_train,
        'x_test': x_test,
        'y_test': y_test,
        'g_test': g_test,
        'x_attacker': x_attacker,
        'y_attacker': y_attacker,
        'g_attacker': g_attacker,
        'x_noise': x_noise,
        'y_noise': y_noise,
        'g_noise': g_noise
    }
    return data


def accuracy(y_true, y_pred):
    if len(y_true) == 0:
        return 0
    return 1 - sum([abs(y_true[i] - y_pred[i]) for i in range(len(y_true))]) / len(y_true)


def cross_entropy(y, t, tol=1e-12):
    pred = np.clip(y, tol, 1 - tol)
    pred_n = np.clip(1 - pred, tol, 1 - tol)
    return - (np.sum(np.multiply(t, np.log(pred)), axis=1) + np.sum(np.multiply(1 - t, np.log(pred_n)), axis=1))


def Equalized_Odds(s, y_pred, y_true):
    y_pred_0_0 = [(1 - y_pred[i]) for i in range(len(s)) if y_true[i] == 0 and s[i] == 0]
    y_pred_1_0 = [(1 - y_pred[i]) for i in range(len(s)) if y_true[i] == 0 and s[i] == 1]
    y_pred_0_1 = [y_pred[i] for i in range(len(s)) if y_true[i] == 1 and s[i] == 0]
    y_pred_1_1 = [y_pred[i] for i in range(len(s)) if y_true[i] == 1 and s[i] == 1]

    loss_0_0 = np.mean(y_pred_0_0)
    loss_1_0 = np.mean(y_pred_1_0)
    loss_0_1 = np.mean(y_pred_0_1)
    loss_1_1 = np.mean(y_pred_1_1)

    return abs(loss_0_0 - loss_1_0), abs(loss_0_1 - loss_1_1)


# # note: write 2y-1 to convert to +/-1 prediction to use Hinge loss
def eval_loss_pfml_equalized_odds(A, b, x, y, g, x_reg, y_reg, g_reg, gamma, al):
    # Calculate the accuracy loss of model (Hinge loss) on feasible poisoning data points
    # l = max(0, 1 - t.y) where t is the target and y is the classifier score
    # 2*y-1 is to convert label from {0,1} to {-1,1}
    s = 1 - (np.dot(x, A) + b) * (2 * y - 1)
    loss = s * (s >= 0)

    # Calculate fairness loss, using approximation method in page 18 [Chang et al]
    # loss2 array contains loss of each feasible poisoning data points
    # loss2_reg array contains loss of each data point in D_c U D_p
    loss2 = (1 - (2 * y - 1) * (np.dot(x, A) + b)) / 2
    loss2_reg = (1 - (2 * y_reg - 1) * (np.dot(x_reg, A) + b)) / 2

    # Indices of points which satisfy the condition of sensitive group and output, there are four possible groups
    idx00 = np.logical_and(g_reg.flatten() == 0, y_reg.flatten() == 0)
    idx01 = np.logical_and(g_reg.flatten() == 0, y_reg.flatten() == 1)
    idx10 = np.logical_and(g_reg.flatten() == 1, y_reg.flatten() == 0)
    idx11 = np.logical_and(g_reg.flatten() == 1, y_reg.flatten() == 1)

    # Calculate sum of fairness loss of each group on D_c U D_p
    s00 = np.sum(loss2_reg[idx00])
    s01 = np.sum(loss2_reg[idx01])
    s10 = np.sum(loss2_reg[idx10])
    s11 = np.sum(loss2_reg[idx11])

    # Number of points in each group on D_c U D_p
    c00 = np.sum(idx00)
    c01 = np.sum(idx01)
    c10 = np.sum(idx10)
    c11 = np.sum(idx11)

    point_acc = []
    point_fair = []

    sum_loss_acc = 0
    sum_loss_fair = 0

    # Iterate over each feasible poisoning data point, check which group it belongs to and calculate the fairness loss
    # of D_c U D_p U {(x,y)}
    for i in range(len(loss)):

        # Introduce temp variables
        s00_temp = s00
        s01_temp = s01
        s10_temp = s10
        s11_temp = s11
        c00_temp = c00
        c01_temp = c01
        c10_temp = c10
        c11_temp = c11

        # Check the group and then add the fairness loss of current feasible poisoning point into its group
        if g[i][0] == 0 and y[i][0] == 0:
            s00_temp += loss2[i][0]
            c00_temp += 1
        elif g[i][0] == 0 and y[i][0] == 1:
            s01_temp += loss2[i][0]
            c01_temp += 1
        elif g[i][0] == 1 and y[i][0] == 0:
            s10_temp += loss2[i][0]
            c10_temp += 1
        elif g[i][0] == 1 and y[i][0] == 1:
            s11_temp += loss2[i][0]
            c11_temp += 1

        # Calculate fairness loss of D_c U D_p U {(x,y)}
        reg = np.abs(s00_temp / c00_temp - s10_temp / c10_temp) + np.abs(s01_temp / c01_temp - s11_temp / c11_temp)

        sum_loss_acc += loss[i][0]
        sum_loss_fair += reg

        point_acc.append(loss[i][0])
        point_fair.append(reg)

        # The total loss with weighted param alpha and Lambda
        loss[i][0] = al * loss[i][0] + (1 - al) * gamma * reg

    return loss, sum_loss_acc / len(loss), sum_loss_fair / len(loss), point_acc, point_fair


def gradient_pfml_equalized_odds(A, b, x_loss, y_loss, x_reg, y_reg, g_reg, L, r, n, t):
    # Calculate the accuracy loss of model (Hinge loss) on feasible poisoning data points
    # l = max(0, 1 - t.y) where t is the target and y is the classifier score
    # 2*y-1 is to convert label from {0,1} to {-1,1}
    # The variable s is used to support the calculation of Gradient
    s = 1 - (np.dot(x_loss, A) + b) * (2 * y_loss - 1)

    # Calculate the Gradient of accuracy loss L
    # dl/dw_i = -t*x_i if 1 - t.y >0 or 0 otherwise
    # Here, we calculate the sum of loss of all feasible poisoning data points
    dfA_loss = np.sum((-x_loss * (2 * y_loss - 1)) * (s >= 0), axis=0).reshape((-1, 1))
    dfb_loss = np.sum(-(2 * y_loss - 1) * (s >= 0))

    # Indices of points which satisfy the condition of sensitive group and output, there are four possible groups
    idx00 = np.logical_and(g_reg.flatten() == 0, y_reg.flatten() == 0)
    idx01 = np.logical_and(g_reg.flatten() == 0, y_reg.flatten() == 1)
    idx10 = np.logical_and(g_reg.flatten() == 1, y_reg.flatten() == 0)
    idx11 = np.logical_and(g_reg.flatten() == 1, y_reg.flatten() == 1)

    # loss2_reg array contains loss of each data point in D_c U D_p
    loss2_reg = (1 - (2 * y_reg - 1) * (np.dot(x_reg, A) + b)) / 2

    # Calculate sum of fairness loss of each group on D_c U D_p
    s00 = np.sum(loss2_reg[idx00])
    s01 = np.sum(loss2_reg[idx01])
    s10 = np.sum(loss2_reg[idx10])
    s11 = np.sum(loss2_reg[idx11])

    # Number of points in each group on D_c U D_p
    c00 = np.sum(idx00)
    c01 = np.sum(idx01)
    c10 = np.sum(idx10)
    c11 = np.sum(idx11)

    # The gradient of approximation function for fairness loss
    dfA_reg_0 = (L * np.sign(s00 / c00 - s10 / c10) * (
            np.mean(-(2 * y_reg[idx00] - 1) * x_reg[idx00] / 2, axis=0) - np.mean(
        -(2 * y_reg[idx10] - 1) * x_reg[idx10] / 2, axis=0))).reshape((-1, 1))
    dfb_reg_0 = L * np.sign(s00 / c00 - s10 / c10) * (
            -sum(2 * y_reg[idx00] - 1) / 2 / len(idx00) - -sum(2 * y_reg[idx10] - 1) / 2 / len(idx10))

    dfA_reg_1 = (L * np.sign(s01 / c01 - s11 / c11) * (
            np.mean(-(2 * y_reg[idx01] - 1) * x_reg[idx01] / 2, axis=0) - np.mean(
        -(2 * y_reg[idx11] - 1) * x_reg[idx11] / 2, axis=0))).reshape((-1, 1))
    dfb_reg_1 = L * np.sign(s01 / c01 - s11 / c11) * (
            -sum(2 * y_reg[idx01] - 1) / 2 / len(idx01) - -sum(2 * y_reg[idx11] - 1) / 2 / len(idx11))

    dA_reg = (2 * A).reshape((-1, 1))

    return {
        'dA': (dfA_loss / (n + t) + dfA_reg_0 + dfA_reg_1 + r * dA_reg).reshape((-1, 1)),
        'db': dfb_loss / (n + t) + dfb_reg_0 + dfb_reg_1
    }


def gradient_train_params_equalized_odds(A, b, x_loss, y_loss, x_reg, y_reg, g_reg, L, r, n):
    # Calculate the accuracy loss of model (Hinge loss) on feasible poisoning data points
    # l = max(0, 1 - t.y) where t is the target and y is the classifier score
    # 2*y-1 is to convert label from {0,1} to {-1,1}
    # The variable s is used to support the calculation of Gradient
    s = 1 - (np.dot(x_loss, A) + b) * (2 * y_loss - 1)

    # Calculate the Gradient of accuracy loss L
    # dl/dw_i = -t*x_i if 1 - t.y >0 or 0 otherwise
    # Here, we calculate the sum of loss of all feasible poisoning data points
    dfA_loss = np.sum((-x_loss * (2 * y_loss - 1)) * (s >= 0), axis=0).reshape((-1, 1))
    dfb_loss = np.sum(-(2 * y_loss - 1) * (s >= 0))

    # Indices of points which satisfy the condition of sensitive group and output, there are four possible groups
    idx00 = np.logical_and(g_reg.flatten() == 0, y_reg.flatten() == 0)
    idx01 = np.logical_and(g_reg.flatten() == 0, y_reg.flatten() == 1)
    idx10 = np.logical_and(g_reg.flatten() == 1, y_reg.flatten() == 0)
    idx11 = np.logical_and(g_reg.flatten() == 1, y_reg.flatten() == 1)

    # loss2_reg array contains loss of each data point in D_c U D_p
    loss2_reg = (1 - (2 * y_reg - 1) * (np.dot(x_reg, A) + b)) / 2

    # Calculate sum of fairness loss of each group on D_c U D_p
    s00 = np.sum(loss2_reg[idx00])
    s01 = np.sum(loss2_reg[idx01])
    s10 = np.sum(loss2_reg[idx10])
    s11 = np.sum(loss2_reg[idx11])

    # Number of points in each group on D_c U D_p
    c00 = np.sum(idx00)
    c01 = np.sum(idx01)
    c10 = np.sum(idx10)
    c11 = np.sum(idx11)

    # The gradient of approximation function for fairness loss
    dfA_reg_0 = (L * np.sign(s00 / c00 - s10 / c10) * (
            np.mean(-(2 * y_reg[idx00] - 1) * x_reg[idx00] / 2, axis=0) - np.mean(
        -(2 * y_reg[idx10] - 1) * x_reg[idx10] / 2, axis=0))).reshape((-1, 1))
    dfb_reg_0 = L * np.sign(s00 / c00 - s10 / c10) * (
            -sum(2 * y_reg[idx00] - 1) / 2 / len(idx00) - -sum(2 * y_reg[idx10] - 1) / 2 / len(idx10))

    dfA_reg_1 = (L * np.sign(s01 / c01 - s11 / c11) * (
            np.mean(-(2 * y_reg[idx01] - 1) * x_reg[idx01] / 2, axis=0) - np.mean(
        -(2 * y_reg[idx11] - 1) * x_reg[idx11] / 2, axis=0))).reshape((-1, 1))
    dfb_reg_1 = L * np.sign(s01 / c01 - s11 / c11) * (
            -sum(2 * y_reg[idx01] - 1) / 2 / len(idx01) - -sum(2 * y_reg[idx11] - 1) / 2 / len(idx11))

    dA_reg = (2 * A).reshape((-1, 1))

    return {
        'dA': (dfA_loss / n + dfA_reg_0 + dfA_reg_1 + r * dA_reg).reshape((-1, 1)),
        'db': dfb_loss / n + dfb_reg_0 + dfb_reg_1
    }


##### compute risk difference
def Demographic_Parity(s, y_pred):
    s_1, s_0 = 0, 0
    y_1_s_0, y_1_s_1 = 0, 0
    for i in range(len(s)):
        if s[i] == 0:
            s_0 += 1
        if s[i] == 1:
            s_1 += 1
        if s[i] == 1 and y_pred[i] == 1:
            y_1_s_1 += 1
        if s[i] == 0 and y_pred[i] == 1:
            y_1_s_0 += 1

    return abs(1.0 * y_1_s_1 / s_1 - 1.0 * y_1_s_0 / s_0)


def eval_loss_pfml_demographic_parity(A, b, x, y, g, x_reg, y_reg, g_reg, gamma, al):
    # Calculate the accuracy loss of model (Hinge loss) on feasible poisoning data points
    # l = max(0, 1 - t.y) where t is the target and y is the classifier score
    # 2*y-1 is to convert label from {0,1} to {-1,1}
    s = 1 - (np.dot(x, A) + b) * (2 * y - 1)
    loss = s * (s >= 0)

    point_acc = []
    point_fair = []

    sum_loss_acc = 0
    sum_loss_fair = 0

    for i in range(len(loss)):
        # Calculate fairness loss for each (x, y) U D_c U D_p

        point = x[i].copy()
        point = point.reshape(-1, len(point))

        X_Dc_Dp_xy = np.append(x_reg, point, axis=0)
        g_Dc_Dp_xy = np.append(g_reg, g[i])
        distance = (np.dot(X_Dc_Dp_xy, A) + b)
        s_bar = np.sum(g_Dc_Dp_xy) / len(g_Dc_Dp_xy)

        s_minus_bar = g_Dc_Dp_xy - s_bar
        s_minus_bar = s_minus_bar.reshape(len(s_minus_bar), -1)
        boundary = np.multiply(s_minus_bar, distance)

        fair_loss = (np.sum(boundary) / len(boundary)) ** 2
        point_acc.append(loss[i][0])
        point_fair.append(fair_loss)

        sum_loss_fair += fair_loss
        sum_loss_acc += loss[i][0]

        loss[i][0] = al * loss[i][0] + (1 - al) * gamma * fair_loss

    return loss, sum_loss_acc / len(loss), sum_loss_fair / len(loss), point_acc, point_fair


def gradient_pfml_demographic_parity(A, b, x_loss, y_loss, x_reg, y_reg, g_reg, L, r, n, t):
    # Calculate the accuracy loss of model (Hinge loss) on feasible poisoning data points
    # l = max(0, 1 - t.y) where t is the target and y is the classifier score
    # 2*y-1 is to convert label from {0,1} to {-1,1}
    # The variable s is used to support the calculation of Gradient
    s = 1 - (np.dot(x_loss, A) + b) * (2 * y_loss - 1)

    # Calculate the Gradient of accuracy loss L
    # dl/dw_i = -t*x_i if 1 - t.y >0 or 0 otherwise
    # Here, we calculate the sum of loss of all feasible poisoning data points
    dfA_pred = np.sum((-x_loss * (2 * y_loss - 1)) * (s >= 0), axis=0).reshape((-1, 1))
    dfb_pred = np.sum(-(2 * y_loss - 1) * (s >= 0))

    # Calculate the gradient of the fairness loss part

    distance = (np.dot(x_reg, A) + b)
    s_bar = np.sum(g_reg) / len(g_reg)
    s_minus_bar = g_reg - s_bar
    s_minus_bar = s_minus_bar.reshape(len(s_minus_bar), -1)
    boundary = np.multiply(s_minus_bar, distance)
    fair_loss = np.sum(boundary) / len(boundary)

    #### for jth dimension of A, the gradient is 2*fair_loss*(sum_{i=1}^n(s_i - s_bar)*x_ij/n)
    new_x = np.multiply(s_minus_bar, x_reg) / len(s_minus_bar)

    dfA_fair = L * 2 * fair_loss * np.sum(new_x, axis=0)
    dfA_fair = dfA_fair.reshape((-1, 1))

    dfb_fair = L * 2 * fair_loss * np.sum(s_minus_bar) / len(s_minus_bar)

    return {
        'dA': (dfA_pred / (n + t) + r * dfA_fair).reshape((-1, 1)),
        'db': dfb_pred / (n + t) + dfb_fair  #
    }


def gradient_train_params_demographic_parity(A, b, x_loss, y_loss, x_reg, y_reg, g_reg, L, r, n):
    # Calculate the accuracy loss of model (Hinge loss) on feasible poisoning data points
    # l = max(0, 1 - t.y) where t is the target and y is the classifier score
    # 2*y-1 is to convert label from {0,1} to {-1,1}
    # The variable s is used to support the calculation of Gradient
    s = 1 - (np.dot(x_loss, A) + b) * (2 * y_loss - 1)

    # Calculate the Gradient of accuracy loss L
    # dl/dw_i = -t*x_i if 1 - t.y >0 or 0 otherwise
    # Here, we calculate the sum of loss of all feasible poisoning data points
    dfA_pred = np.sum((-x_loss * (2 * y_loss - 1)) * (s >= 0), axis=0).reshape((-1, 1))
    dfb_pred = np.sum(-(2 * y_loss - 1) * (s >= 0))

    # Calculate the gradient of the fairness loss part

    distance = (np.dot(x_reg, A) + b)
    s_bar = np.sum(g_reg) / len(g_reg)
    s_minus_bar = g_reg - s_bar
    s_minus_bar = s_minus_bar.reshape(len(s_minus_bar), -1)
    boundary = np.multiply(s_minus_bar, distance)
    fair_loss = np.sum(boundary) / len(boundary)

    #### for jth dimension of A, the gradient is 2*fair_loss*(sum_{i=1}^n(s_i - s_bar)*x_ij/n)
    new_x = np.multiply(s_minus_bar, x_reg) / len(s_minus_bar)

    dfA_fair = L * 2 * fair_loss * np.sum(new_x, axis=0)
    dfA_fair = dfA_fair.reshape((-1, 1))

    dfb_fair = L * 2 * fair_loss * np.sum(s_minus_bar) / len(s_minus_bar)

    return {
        'dA': (dfA_pred / n + r * dfA_fair).reshape((-1, 1)),
        'db': dfb_pred / n + dfb_fair  #
    }


##### compute equal opportunity
def Equal_Opportunity(s, y_pred, y_true):
    y_pred_0_1 = [y_pred[i] for i in range(len(s)) if y_true[i] == 1 and s[i] == 0]
    y_pred_1_1 = [y_pred[i] for i in range(len(s)) if y_true[i] == 1 and s[i] == 1]

    loss_0_1 = np.mean(y_pred_0_1)
    loss_1_1 = np.mean(y_pred_1_1)

    return abs(loss_0_1 - loss_1_1)


def eval_loss_pfml_equal_opportunity(A, b, x, y, g, x_reg, y_reg, g_reg, gamma, al):
    # Calculate the accuracy loss of model (Hinge loss) on feasible poisoning data points
    # l = max(0, 1 - t.y) where t is the target and y is the classifier score
    # 2*y-1 is to convert label from {0,1} to {-1,1}
    s = 1 - (np.dot(x, A) + b) * (2 * y - 1)
    loss = s * (s >= 0)

    # Calculate fairness loss, using approximation method in page 18 [Chang et al]
    # loss2 array contains loss of each feasible poisoning data points
    # loss2_reg array contains loss of each data point in D_c U D_p
    loss2 = (1 - (2 * y - 1) * (np.dot(x, A) + b)) / 2
    loss2_reg = (1 - (2 * y_reg - 1) * (np.dot(x_reg, A) + b)) / 2

    # Indices of points which satisfy the condition of sensitive group and output, there are four possible groups
    idx00 = np.logical_and(g_reg.flatten() == 0, y_reg.flatten() == 0)
    idx01 = np.logical_and(g_reg.flatten() == 0, y_reg.flatten() == 1)
    idx10 = np.logical_and(g_reg.flatten() == 1, y_reg.flatten() == 0)
    idx11 = np.logical_and(g_reg.flatten() == 1, y_reg.flatten() == 1)

    # Calculate sum of fairness loss of each group on D_c U D_p
    s00 = np.sum(loss2_reg[idx00])
    s01 = np.sum(loss2_reg[idx01])
    s10 = np.sum(loss2_reg[idx10])
    s11 = np.sum(loss2_reg[idx11])

    # Number of points in each group on D_c U D_p
    c00 = np.sum(idx00)
    c01 = np.sum(idx01)
    c10 = np.sum(idx10)
    c11 = np.sum(idx11)

    point_acc = []
    point_fair = []

    sum_loss_acc = 0
    sum_loss_fair = 0

    # Iterate over each feasible poisoning data point, check which group it belongs to and calculate the fairness loss
    # of D_c U D_p U {(x,y)}
    for i in range(len(loss)):

        # Introduce temp variables
        s00_temp = s00
        s01_temp = s01
        s10_temp = s10
        s11_temp = s11
        c00_temp = c00
        c01_temp = c01
        c10_temp = c10
        c11_temp = c11

        # Check the group and then add the fairness loss of current feasible poisoning point into its group
        if g[i][0] == 0 and y[i][0] == 0:
            s00_temp += loss2[i][0]
            c00_temp += 1
        elif g[i][0] == 0 and y[i][0] == 1:
            s01_temp += loss2[i][0]
            c01_temp += 1
        elif g[i][0] == 1 and y[i][0] == 0:
            s10_temp += loss2[i][0]
            c10_temp += 1
        elif g[i][0] == 1 and y[i][0] == 1:
            s11_temp += loss2[i][0]
            c11_temp += 1

        # Calculate fairness loss of D_c U D_p U {(x,y)}
        reg = np.abs(s01_temp / c01_temp - s11_temp / c11_temp)

        sum_loss_acc += loss[i][0]
        sum_loss_fair += reg

        point_acc.append(loss[i][0])
        point_fair.append(reg)

        # The total loss with weighted param alpha and gamma
        loss[i][0] = al * loss[i][0] + (1 - al) * gamma * reg

    return loss, sum_loss_acc / len(loss), sum_loss_fair / len(loss), point_acc, point_fair


def gradient_pfml_equal_opportunity(A, b, x_loss, y_loss, x_reg, y_reg, g_reg, L, r, n, t):
    # Calculate the accuracy loss of model (Hinge loss) on feasible poisoning data points
    # l = max(0, 1 - t.y) where t is the target and y is the classifier score
    # 2*y-1 is to convert label from {0,1} to {-1,1}
    # The variable s is used to support the calculation of Gradient
    s = 1 - (np.dot(x_loss, A) + b) * (2 * y_loss - 1)

    # Calculate the Gradient of accuracy loss L
    # dl/dw_i = -t*x_i if 1 - t.y >0 or 0 otherwise
    # Here, we calculate the sum of loss of all feasible poisoning data points
    dfA_loss = np.sum((-x_loss * (2 * y_loss - 1)) * (s >= 0), axis=0).reshape((-1, 1))
    dfb_loss = np.sum(-(2 * y_loss - 1) * (s >= 0))

    # Indices of points which satisfy the condition of sensitive group and output, there are four possible groups
    idx01 = np.logical_and(g_reg.flatten() == 0, y_reg.flatten() == 1)
    idx11 = np.logical_and(g_reg.flatten() == 1, y_reg.flatten() == 1)

    # loss2_reg array contains loss of each data point in D_c U D_p
    loss2_reg = (1 - (2 * y_reg - 1) * (np.dot(x_reg, A) + b)) / 2

    # Calculate sum of fairness loss of each group on D_c U D_p
    s01 = np.sum(loss2_reg[idx01])
    s11 = np.sum(loss2_reg[idx11])

    # Number of points in each group on D_c U D_p
    c01 = np.sum(idx01)
    c11 = np.sum(idx11)

    # The gradient of approximation function for fairness loss
    dfA_reg_1 = (L * np.sign(s01 / c01 - s11 / c11) * (
            np.mean(-(2 * y_reg[idx01] - 1) * x_reg[idx01] / 2, axis=0) - np.mean(
        -(2 * y_reg[idx11] - 1) * x_reg[idx11] / 2, axis=0))).reshape((-1, 1))
    dfb_reg_1 = L * np.sign(s01 / c01 - s11 / c11) * (
            -sum(2 * y_reg[idx01] - 1) / 2 / len(idx01) - -sum(2 * y_reg[idx11] - 1) / 2 / len(idx11))

    dA_reg = (2 * A).reshape((-1, 1))

    return {
        'dA': (dfA_loss / (n + t) + dfA_reg_1 + r * dA_reg).reshape((-1, 1)),
        'db': dfb_loss / (n + t) + dfb_reg_1
    }


def gradient_train_params_equal_opportunity(A, b, x_loss, y_loss, x_reg, y_reg, g_reg, L, r, n):
    # Calculate the accuracy loss of model (Hinge loss) on feasible poisoning data points
    # l = max(0, 1 - t.y) where t is the target and y is the classifier score
    # 2*y-1 is to convert label from {0,1} to {-1,1}
    # The variable s is used to support the calculation of Gradient
    s = 1 - (np.dot(x_loss, A) + b) * (2 * y_loss - 1)

    # Calculate the Gradient of accuracy loss L
    # dl/dw_i = -t*x_i if 1 - t.y >0 or 0 otherwise
    # Here, we calculate the sum of loss of all feasible poisoning data points
    dfA_loss = np.sum((-x_loss * (2 * y_loss - 1)) * (s >= 0), axis=0).reshape((-1, 1))
    dfb_loss = np.sum(-(2 * y_loss - 1) * (s >= 0))

    # Indices of points which satisfy the condition of sensitive group and output, there are four possible groups
    idx01 = np.logical_and(g_reg.flatten() == 0, y_reg.flatten() == 1)
    idx11 = np.logical_and(g_reg.flatten() == 1, y_reg.flatten() == 1)

    # loss2_reg array contains loss of each data point in D_c U D_p
    loss2_reg = (1 - (2 * y_reg - 1) * (np.dot(x_reg, A) + b)) / 2

    # Calculate sum of fairness loss of each group on D_c U D_p
    s01 = np.sum(loss2_reg[idx01])
    s11 = np.sum(loss2_reg[idx11])

    # Number of points in each group on D_c U D_p
    c01 = np.sum(idx01)
    c11 = np.sum(idx11)

    # The gradient of approximation function for fairness loss
    dfA_reg_1 = (L * np.sign(s01 / c01 - s11 / c11) * (
            np.mean(-(2 * y_reg[idx01] - 1) * x_reg[idx01] / 2, axis=0) - np.mean(
        -(2 * y_reg[idx11] - 1) * x_reg[idx11] / 2, axis=0))).reshape((-1, 1))
    dfb_reg_1 = L * np.sign(s01 / c01 - s11 / c11) * (
            -sum(2 * y_reg[idx01] - 1) / 2 / len(idx01) - -sum(2 * y_reg[idx11] - 1) / 2 / len(idx11))

    dA_reg = (2 * A).reshape((-1, 1))

    return {
        'dA': (dfA_loss / n + dfA_reg_1 + r * dA_reg).reshape((-1, 1)),
        'db': dfb_loss / n + dfb_reg_1
    }

def l2_norm(v):
    return np.linalg.norm(v)
