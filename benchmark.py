from sklearn.linear_model import LogisticRegression
from models import GrapeModule
from dataprocessing import load_data
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


def train_model(config: ModelConfig, X, y, X_val, y_val):
    model = config.model(**config.model_kwargs)
    model, train_loss = config.train_procedure(model, X, y, X_val, y_val)
    print(f"Trained model {config.name} - train loss: {train_loss}")
    return model


def run_experiment(config: ExperimentConfig):
    X_train, y_train, X_test, y_test = load_data(config)

    metrics = dict()
    for name, model in config.models.items():
        trained_model = train_model(model, X_train, y_train, X_test, y_test)
        metrics[model.name] = model.eval_procedure(trained_model, X_test, y_test)

    return metrics


if __name__=="__main__":
    n_genes = 100
    config = ExperimentConfig(
        n_genes=n_genes,
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
                    'emb_dim': 4,
                    'n_layers': 2,
                    'edge_dim': 1,
                    'out_dim': 19,
                    'n_genes': 3451,
                },
            ),
            # 'ogre': ModelConfig(
            #     name='OGRE',
            #     model=OgreModule,
            #     train_procedure=train_grape,
            #     eval_procedure=eval_grape,
            #     n_genes=n_genes,
            #     model_kwargs={
            #         'emb_dim': 4,
            #         'n_layers': 2,
            #         'edge_dim': 1,
            #         'n_pathways': ...,
            #         'out_dim': 19,
            #         'n_genes': n_genes,
            #     },
            # )
        }

    )
    print(run_experiment(config))
