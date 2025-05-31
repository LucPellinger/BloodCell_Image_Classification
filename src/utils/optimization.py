# utils/optimization.py


import optuna
import os
import sys

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.integration import TFKerasPruningCallback
from optuna.pruners import MedianPruner

import tensorflow as tf

from utils.train_utils import performance_plot
from app.components.path_utils import get_project_root


class BayesianOptimizer:
    def __init__(self, model_class):
        self.model_class = model_class
        self.db_path = "sqlite:///" + os.path.abspath(os.path.join(get_project_root(), "assets", "optuna_study.db"))
        self.study = None

    def _objective(self, trial, **kwargs):
        # Suggest hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        num_dense_layers = trial.suggest_int('num_dense_layers', 1, 2)
        neurons_dense = trial.suggest_int('neurons_dense', 128, 256)
        dropout_rate = trial.suggest_categorical('dropout_rate', [0.5, None])
        normalization = trial.suggest_categorical('normalization', [True, False])
        l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True)

        # Initialize model
        model = self.model_class(
            model_name=kwargs['model_name'],
            num_classes=kwargs['num_classes'],
            input_shape=(kwargs['img_height'], kwargs['img_width'], kwargs['num_channels'])
        )

        model.build_pretrained(
            base_model_name=kwargs['model_name'],
            num_dense_layers=num_dense_layers,
            neurons_dense=neurons_dense,
            dropout_rate=dropout_rate,
            normalization=normalization,
            l2_reg=l2_reg,
            augmentation=True
        )

        trial_log_dir = os.path.join(
            get_project_root(), "assets", "tf_logs_optuna",  # base log dir
            kwargs['model_name'],                    # e.g., VGG16
            f"trial_{trial.number}"                  # trial ID
        )
        os.makedirs(trial_log_dir, exist_ok=True)

        pruning_callback = TFKerasPruningCallback(trial, 'val_accuracy')

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        model.train(
            train_ds=kwargs['train_ds'],
            val_ds=kwargs['val_ds'],
            epochs=kwargs['num_epochs'],
            early_stopping=True,
            tensorboard_logdir=trial_log_dir,
            plot=False,
            additional_callbacks=[pruning_callback]
        )

        results = model.model.evaluate(kwargs['val_ds'], verbose=0)
        return results[1]  # Return validation accuracy

    def run(self, train_ds, val_ds, test_ds, model_name, num_classes, img_height, img_width, num_channels, num_epochs, n_trials=50):
        self.study = optuna.create_study(
            study_name=f"BayesOpt_{model_name}",
            direction='maximize',
            storage=self.db_path,
            load_if_exists=True,
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0)
        )

        self.study.optimize(
            lambda trial: self._objective(
                trial,
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
                model_name=model_name,
                num_classes=num_classes,
                img_height=img_height,
                img_width=img_width,
                num_channels=num_channels,
                num_epochs=num_epochs
            ),
            n_trials=n_trials
        )

        print("Best hyperparameters:", self.study.best_params)
        return self.study.best_trial



    def visualize(self):
        if self.study:
            plot_optimization_history(self.study).show()
            plot_param_importances(self.study).show()
        else:
            print("No study found to visualize.")
