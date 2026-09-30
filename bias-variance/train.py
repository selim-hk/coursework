import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
from sklearn.metrics import mean_squared_error

from config import CONFIG
from data import train_xs, train_ys
from features import fourier_features, gaussian_features, identical_features, polynomial_features

length = len(train_xs)

ratio_cands = [(0.5, 0.3, 0.2), (0.6, 0.2, 0.2), (0.8, 0.1, 0.1)]

train_length = int(length * ratio_cands[1][0])
val_length = int(length * ratio_cands[1][1])
test_length = length - train_length - val_length

train, val, test = np.split(train_xs, [train_length, train_length + val_length])
train_y, val_y, test_y = np.split(train_ys, [train_length, train_length + val_length])

full_data = [train, val, test]


results_df = pd.DataFrame()

for feature_func in CONFIG["feature_cands"]:
    for num_features in CONFIG["num_features"]:
        for lam in CONFIG["lamda_cands"]:

            if feature_func == "polynomial":
                train_curr, val_curr = polynomial_features(train, num_features), polynomial_features(val, num_features)
            elif feature_func == "fourier":
                train_curr, val_curr = fourier_features(train, num_features), fourier_features(val, num_features)
            elif feature_func == "gaussian":
                train_curr, val_curr = gaussian_features(train, num_features), gaussian_features(val, num_features)
            elif feature_func == "identical":
                train_curr, val_curr = identical_features(train), identical_features(val)

            model = sklearn.linear_model.Ridge(alpha = lam)
            model.fit(train_curr, train_y)
            val_pred = model.predict(val_curr)

            mse = mean_squared_error(val_y, val_pred)

            df = pd.DataFrame({
                "feature_func": [feature_func],
                "num_features": [num_features],
                "lam": [lam],
                "mse": [mse]
            })

            results_df = pd.concat([results_df, df], ignore_index=True)


lambda_column = results_df['lam']
val_mse_column = results_df['mse']

types = results_df['feature_func'].unique()
num_feats = results_df['num_features'].unique()

fig, axes = plt.subplots(3, 2, figsize=(20, 16))
axes = axes.flatten()

for i, n in enumerate(num_feats):
    subset = results_df[results_df['num_features'] == n]
    ax = axes[i]
    for t in types:
        subset_t = subset[subset['feature_func'] == t]
        ax.plot(subset_t['lam'], subset_t['mse'], label=t)
    ax.set_xlabel('Lambda')
    ax.set_ylabel('Validation MSE')
    ax.set_title(f'Validation MSE vs Lambda\n(num_features={n} and ratio={ratio_cands[1]})')
    ax.legend()
    ax.set_xscale('log')

plt.tight_layout()
plt.show()

row = results_df.loc[results_df['mse'].idxmin()]
print(row)

best_hyperparams = {
    "feature_func": "fourier",
    "num_features": 25,
    "lam": 0.05
}
train = np.concatenate([train, val], axis=0)
train_y = np.concatenate([train_y, val_y], axis=0)

train, test = fourier_features(train, best_hyperparams["num_features"]), fourier_features(test, best_hyperparams["num_features"])
my_model = sklearn.linear_model.Ridge(alpha=best_hyperparams["lam"])
my_model.fit(train, train_y)
test_pred = my_model.predict(test)
test_mse = mean_squared_error(test_y, test_pred)
print(test_mse)
