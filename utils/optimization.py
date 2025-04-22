import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import tensorflow as tf
from utils.archive.train_utils import performance_plot

class BayesianOptimizer:
    def __init__(self, model_class, db_path="sqlite:///assets/optuna_study.db"):
        self.model_class = model_class
        self.db_path = db_path
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
            l2_reg=l2_reg
        )

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        model.train(
            train_ds=kwargs['train_ds'],
            val_ds=kwargs['val_ds'],
            epochs=kwargs['num_epochs'],
            early_stopping=True,
            tensorboard_logdir=None
        )

        results = model.model.evaluate(kwargs['val_ds'], verbose=0)
        return results[1]  # Return validation accuracy

    def run(self, train_ds, val_ds, test_ds, model_name, num_classes, img_height, img_width, num_channels, num_epochs, n_trials=50):
        self.study = optuna.create_study(
            study_name=f"BayesOpt_{model_name}",
            direction='maximize',
            storage=self.db_path,
            load_if_exists=True
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
