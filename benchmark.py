from sklearn.linear_model import LogisticRegression
from models import GrapeModule, OgreModule
from dataprocessing import load_data, random_p
from training import (
    train_grape, 
    train_logistic_regression,
    eval_grape,
    eval_logistic_regression
)

from dataclasses import dataclass

from typing import Dict, Any, Callable


@dataclass
class ModelConfig:
    name: str
    model: Callable
    train_procedure: Callable
    eval_procedure: Callable
    model_kwargs: Dict


@dataclass
class ExperimentConfig:
    models: Dict[str, ModelConfig]
    n_genes: int
    use_pathways: int


def train_model(config: ModelConfig, X, y, X_val, y_val, P, n_genes):
    if config.name == "LogisticRegression":
        model = config.model(**config.model_kwargs)
    else:
        model = config.model(**config.model_kwargs, P=P, n_genes=n_genes)
    model, train_loss = config.train_procedure(model, X, y, X_val, y_val, P)
    print(f"Trained model {config.name} - train loss: {train_loss}")
    return model


def run_experiment(config: ExperimentConfig):
    X_train, y_train, X_test, y_test, pathways = load_data(config)
    print(config.n_genes)

    metrics = dict()
    for name, model in config.models.items():
        print(f"training on data of shape {X_train.shape}")
        if name == 'FullOGRE':
            P = random_p(pathways)
        else:
            P = pathways

        trained_model = train_model(model, X_train, y_train, X_test, y_test, P, config.n_genes)
        metrics[model.name] = model.eval_procedure(trained_model, X_test, y_test, P, config.n_genes)

    return metrics


def run_gene_count_experiment():
    results = []
    for n_genes in range(100, 1550, 10):
        config = ExperimentConfig(
            n_genes=n_genes,
            use_pathways = True,
            models = {
                'logistic_regression': ModelConfig(
                    name='LogisticRegression',
                    model=LogisticRegression,
                    train_procedure=train_logistic_regression,
                    eval_procedure=eval_logistic_regression,
                    model_kwargs=dict(),
                ),
                'grape': ModelConfig(
                    name='GRAPE',
                    model=GrapeModule,
                    train_procedure=train_grape,
                    eval_procedure=eval_grape,
                    model_kwargs={
                        'emb_dim': 20,
                        'n_layers': 2,
                        'edge_dim': 1,
                        'out_dim': 19,
                    },
                ),
                'ogre': ModelConfig(
                    name='OGRE',
                    model=OgreModule,
                    train_procedure=train_grape,
                    eval_procedure=eval_grape,
                    model_kwargs={
                        'emb_dim': 20,
                        'n_layers': 2,
                        'edge_dim': 1,
                        'out_dim': 19,
                    },
                ),
                'FullOGRE': ModelConfig(
                    name='FullOGRE',
                    model=OgreModule,
                    train_procedure=train_grape,
                    eval_procedure=eval_grape,
                    model_kwargs={
                        'emb_dim': 20,
                        'n_layers': 2,
                        'edge_dim': 1,
                        'out_dim': 19,
                    },
                )
            }
        )
        res = run_experiment(config)
        results.append({**res, **{'n_genes': n_genes}})

    return results


if __name__ == "__main__":
    results = run_gene_count_experiment()
    import pandas as pd
    df = pd.DataFrame(results)
    print(df)
    df.to_csv('first_final_test.csv', index=False)
