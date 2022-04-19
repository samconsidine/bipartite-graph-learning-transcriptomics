from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel:
    def __init__(self, **log_reg_kwargs):
        self.model = LogisticRegression(**log_reg_kwargs)
        self.log_reg_kwargs = log_reg_kwargs

    def __repr__(self):
        return str(log_reg_kwargs)


def train_logistic_regression(
        model: LogisticRegression,
        data: Andata,
        **pca_kwargs
    ) -> LogisticRegression:

    dim_reduction = PCA(**pca_kwags)
    inputs = dim_reduction.fit_transform(data.values)

    model.fit(X, y)
    return model


def score_logistic_regression(model: LogisticRegression, data: Andata) -> float:
    return model.score(X, y)

