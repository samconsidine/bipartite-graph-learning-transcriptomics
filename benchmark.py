from sklearn.linear_model import LogisticRegression
from models import GrapeModule, PathwaysModule
from dataprocessing import load_data, random_p
from training import (
    train_grape, 
    train_logistic_regression,
    eval_grape,
    eval_logistic_regression
)

from dataclasses import dataclass
from sys import argv
import os

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
    name: str
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
        if name == 'random_pathways':
            P = random_p(pathways)
        else:
            P = pathways

        trained_model = train_model(model, X_train, y_train, X_test, y_test, P, config.n_genes)
        metrics[model.name] = model.eval_procedure(trained_model, X_test, y_test, P, config.n_genes)

    return metrics


def run_gene_count_experiment():
    results = []
    for n_genes in range(100, 150, 50):
        config = ExperimentConfig(
            name='all_models',
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
                'pathways': ModelConfig(
                    name='pathways',
                    model=PathwaysModule,
                    train_procedure=train_grape,
                    eval_procedure=eval_grape,
                    model_kwargs={
                        'emb_dim': 20,
                        'n_layers': 2,
                        'edge_dim': 1,
                        'out_dim': 19,
                    },
                ),
                'random_pathways': ModelConfig(
                    name='random_pathways',
                    model=PathwaysModule,
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

    import pandas as pd
    df = pd.DataFrame(results)
    print(df)
    if not os.path.exists('experiments'):
        os.mkdir('experiments')

    df.to_csv(f'experiments/{config.name}.csv', index=False)
    return results


if __name__ == "__main__":
    results = run_gene_count_experiment()
