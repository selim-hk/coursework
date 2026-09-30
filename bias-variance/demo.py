import numpy as np
import pandas as pd
import plotnine as gg
import sklearn.linear_model

gg.theme_set(gg.theme_bw());

num_data = 100
noise_sigma = 0.2

train_xs = np.random.uniform(-np.pi, np.pi, size=num_data)
train_ys = np.sin(train_xs) + noise_sigma * np.random.randn(num_data)

print(train_xs)

df = pd.DataFrame({'x': train_xs,
                   'y': train_ys,
                   'h': np.sin(train_xs),
                   'std': noise_sigma,
                   })

p = (gg.ggplot(df)
     + gg.aes(x='x', y='h')
     + gg.geom_point(gg.aes(y='y'))
     + gg.geom_line(color='green', size=2)
     + gg.geom_ribbon(gg.aes(ymin='h - std', ymax='h + std'), alpha=0.2)
     + gg.geom_ribbon(gg.aes(ymin='h - 2 * std', ymax='h + 2 * std'), alpha=0.1)
     + gg.ggtitle(r"$p(x, y)$: Mean in green, $\sigma$ and 2$\sigma$ intervals in grey")
     + gg.xlab(r"$x$")
     + gg.ylab(r"$y$")
)
p.show()

num_train = 25
num_models = 20
num_features = 24

polynomial_features = np.array([train_xs**n for n in range(num_features)]).T
fourier_features = np.array([np.sin(2*np.pi*n*train_xs / num_features) for n in range(num_features)]).T
gaussian_features = np.array([np.exp(-(train_xs - (2*np.pi*x/num_features - np.pi))**2) for x in range(num_features)]).T

features = gaussian_features
results_df = pd.DataFrame()
for lam in [1e-3, 1e-1, 1e0, 1e1]:
    for idx in range(num_models):
        train_idx = np.random.randint(len(features), size=num_train)
        train_features = features[train_idx]
        train_targets = train_ys[train_idx]

        model = sklearn.linear_model.Ridge(alpha=lam)
        model.fit(train_features, train_targets)

        df = pd.DataFrame({
            'x': train_xs,
            'y': train_ys,
            'h': np.sin(train_xs),
            'f': model.predict(features),
            'idx': [idx] * num_data,
            'lam': lam,
        })
        results_df = pd.concat([results_df, df])

fit_plot = (gg.ggplot(results_df)
            + gg.aes(x='x')

            + gg.geom_line(gg.aes(y='f', group='idx'), color='red', alpha=0.5)
            + gg.facet_wrap('lam', labeller='label_both', ncol=4)
            + gg.coord_cartesian(ylim=[-1.1, 1.1])
            + gg.theme(figure_size=(12, 3))
            + gg.xlab(r'$x$')
            + gg.ylab(r'$f(x)$')
)
fit_plot.show()

mean_df = results_df.groupby(['x', 'lam', 'y', 'h']).agg({'f': "mean"}).reset_index()

mean_plot = (gg.ggplot(mean_df)
             + gg.aes(x='x', y='f')
             + gg.geom_line(gg.aes(y='h'), color='green')
             + gg.geom_line(color='red')
             + gg.facet_wrap('lam', labeller='label_both', ncol=4)
             + gg.theme(figure_size=(12, 3))
             + gg.xlab(r'$x$')
             + gg.ylab(r'$\mathbb{E}_\mathcal{D}f(x)$')
)
mean_plot.show()
