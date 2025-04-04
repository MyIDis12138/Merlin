import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None


class ClinicalDataSVM:
    """A class to handle loading, processing, and modeling clinical data using SVM."""

    def __init__(
        self,
        clinical_data_path,
        target_column,
        filter_dict=None,
        exclude_columns=None,
        test_size=0.20,
        n_folds=5,
        n_optuna_trials=50,
        random_state=42,
    ):
        """
        Initialize the model with configuration parameters.

        Args:
            clinical_data_path (str): Path to the Excel file with clinical data.
            target_column (tuple): MultiIndex column tuple for the target variable.
            filter_dict (dict, optional): Dictionary to filter columns by level.
            exclude_columns (list, optional): List of specific columns to exclude.
            test_size (float): Proportion of data to use for testing.
            n_folds (int): Number of cross-validation folds.
            random_state (int): Random seed for reproducibility.
        """
        self.clinical_data_path = clinical_data_path
        self.target_column = target_column
        self.filter_dict = filter_dict
        self.exclude_columns = exclude_columns
        self.test_size = test_size
        self.n_folds = n_folds
        self.n_optuna_trials = n_optuna_trials
        self.random_state = random_state

        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.final_model = None
        self.best_params = None

    def read_data(self):
        """
        Load clinical data from an Excel file with a multi-index header.
        """
        try:
            self.df = pd.read_excel(self.clinical_data_path, header=[0, 1, 2])

            new_cols = []
            for col in self.df.columns:
                if isinstance(col, tuple) and isinstance(col[-1], str) and "Unnamed" in col[-1]:
                    new_cols.append(col[:-1] + ("",))
                else:
                    new_cols.append(col)

            self.df.columns = pd.MultiIndex.from_tuples(new_cols)
            logger.info(f"Successfully loaded clinical data from {self.clinical_data_path}")
            logger.info(f"Data shape: {self.df.shape}")
            return True

        except FileNotFoundError:
            logger.error(f"Error: File not found at {self.clinical_data_path}")
            return False
        except Exception as e:
            logger.error(f"Failed to load clinical data: {e}")
            return False

    def prepare_data(self):
        """
        Clean data, handle missing values, prepare features, and filter columns.
        """
        if self.df is None:
            logger.error("No data loaded. Call read_data() first.")
            return False

        if self.target_column in self.df.columns:
            target_series = self.df[self.target_column].copy()
            has_target = True
        else:
            has_target = False
            logger.warning(f"Target column {self.target_column} not found in original data.")

        if self.filter_dict is not None or self.exclude_columns is not None:
            df_processed = self.filter_features()
        else:
            df_processed = self.df.copy()

        if has_target and self.target_column not in df_processed.columns:
            logger.info(f"Re-adding target column {self.target_column} after filtering.")
            df_processed[self.target_column] = target_series

        if self.target_column not in df_processed.columns:
            logger.error(f"Error: Target column {self.target_column} not found in the data.")
            logger.info(f"Available columns: {df_processed.columns}")
            return False

        initial_rows = len(df_processed)
        df_processed.dropna(subset=[self.target_column], inplace=True)
        rows_after_dropna = len(df_processed)
        logger.info(f"Removed {initial_rows - rows_after_dropna} rows with missing target values.")

        if df_processed.empty:
            logger.error("Error: No data remaining after removing missing target values.")
            return False

        try:
            df_processed[self.target_column] = df_processed[self.target_column].astype(int)
            logger.info(f"Target column distribution:\n{df_processed[self.target_column].value_counts(normalize=True)}")
        except Exception as e:
            logger.error(f"Error converting target column to integer: {e}")
            return False

        self.y = df_processed[self.target_column]
        self.X = df_processed.drop(columns=[self.target_column])

        original_X_columns = self.X.columns.copy()

        flat_X_columns = ["_".join(filter(None, map(str, col))).strip("_") for col in original_X_columns]
        self.X.columns = flat_X_columns

        self._process_features(df_processed, original_X_columns)

        return True

    def filter_features(self):
        """
        Filter features based on multi-level headers.
        """
        if self.df is None:
            logger.error("No data loaded. Call read_data() first.")
            return None

        filtered_df = self.df.copy()

        all_columns = filtered_df.columns
        columns_to_drop = []

        logger.info(f"Column type: {type(all_columns)}")
        logger.info(f"Columns nlevels: {all_columns.nlevels}")
        if len(all_columns) > 0:
            logger.info(f"First column sample: {all_columns[0]}")

        if self.filter_dict is not None:
            logger.info(f"Filtering columns based on level criteria: {self.filter_dict}")

            if isinstance(all_columns, pd.MultiIndex):
                for level, values_to_filter in self.filter_dict.items():
                    for value in values_to_filter:
                        level_matches = all_columns[all_columns.get_level_values(level) == value]
                        logger.info(f"Found {len(level_matches)} columns with '{value}' at level {level}")
                        columns_to_drop.extend(level_matches.tolist())
            else:
                logger.warning("Columns are not MultiIndex, applying simple filtering")
                for level, values_to_filter in self.filter_dict.items():
                    if level != 0:
                        logger.warning(f"Cannot filter on level {level} as columns are not MultiIndex")
                        continue
                    columns_to_drop.extend([col for col in all_columns if col in values_to_filter])

        if self.exclude_columns is not None:
            logger.info(f"Excluding specific columns: {self.exclude_columns}")
            for col in self.exclude_columns:
                if col in all_columns:
                    columns_to_drop.append(col)

        columns_to_drop = list(set(columns_to_drop))

        if not columns_to_drop:
            logger.warning("No columns matched the filtering criteria")
        else:
            logger.info(f"Total columns to drop: {len(columns_to_drop)}")
            if columns_to_drop:
                sample_size = min(5, len(columns_to_drop))
                logger.info(f"Sample columns being dropped (first {sample_size}): {columns_to_drop[:sample_size]}")

        if columns_to_drop:
            filtered_df = filtered_df.drop(columns=columns_to_drop)

        logger.info(f"Original shape: {self.df.shape}, Filtered shape: {filtered_df.shape}")

        return filtered_df

    def _process_features(self, df_processed, original_X_columns):
        """
        Helper method to process features: identify types, handle missing values, encode categorical.

        Args:
            df_processed (pd.DataFrame): Processed dataframe with target.
            original_X_columns (pd.Index): Original MultiIndex columns.
        """
        numerical_cols = []
        categorical_cols = []

        col_mapping_flat_to_original = {flat: orig for flat, orig in zip(self.X.columns, original_X_columns)}

        for i, col in enumerate(self.X.columns):
            original_col_tuple = original_X_columns[i]

            self.X[col] = pd.to_numeric(self.X[col], errors="coerce")

            original_col_data_series = df_processed[original_col_tuple]

            if pd.api.types.is_numeric_dtype(self.X[col]) and not self.X[col].isnull().all():
                numerical_cols.append(col)
            else:
                categorical_cols.append(col)
                self.X[col] = original_col_data_series.astype(str)

        logger.info(f"Identified {len(numerical_cols)} numerical columns.")
        logger.info(f"Identified {len(categorical_cols)} categorical columns.")

        if numerical_cols:
            num_imputer = SimpleImputer(strategy="median")
            self.X[numerical_cols] = num_imputer.fit_transform(self.X[numerical_cols])
            logger.info("Imputed missing values in numerical columns using median.")

        if categorical_cols:
            cat_imputer = SimpleImputer(strategy="constant", fill_value="Missing")
            self.X[categorical_cols] = cat_imputer.fit_transform(self.X[categorical_cols])
            logger.info("Imputed missing values in categorical columns with 'Missing'.")

            self.X = pd.get_dummies(self.X, columns=categorical_cols, drop_first=True, dummy_na=False, dtype=int)
            logger.info("Applied one-hot encoding to categorical columns.")
            logger.info(f"Data shape after encoding: {self.X.shape}")

            logger.info("Cleaning column names for SVM compatibility...")
            original_cols = self.X.columns.tolist()
            self.X.columns = self.X.columns.str.replace("[\[\]<]", "_", regex=True)
            cleaned_cols = self.X.columns.tolist()

            changed_cols = [(orig, clean) for orig, clean in zip(original_cols, cleaned_cols) if orig != clean]
            if changed_cols:
                logger.info(f"Cleaned {len(changed_cols)} column names.")

        non_numeric_cols = self.X.select_dtypes(exclude=np.number).columns
        if len(non_numeric_cols) > 0:
            logger.warning(f"Found {len(non_numeric_cols)} non-numeric columns after processing. Attempting final conversion.")
            for col in non_numeric_cols:
                try:
                    self.X[col] = pd.to_numeric(self.X[col], errors="coerce")
                except Exception as e:
                    logger.error(f"Could not convert column {col} to numeric: {e}")

            final_numerical_cols = self.X.select_dtypes(include=np.number).columns
            if self.X.isnull().any().any():
                final_num_imputer = SimpleImputer(strategy="median")
                self.X[final_numerical_cols] = final_num_imputer.fit_transform(self.X[final_numerical_cols])

        logger.info(f"Final feature shape: {self.X.shape}")

    def split_data(self):
        """
        Split data into training and test sets.
        """
        if self.X is None or self.y is None:
            logger.error("Features and target not prepared. Call prepare_data() first.")
            return False

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state, stratify=self.y
        )

        # SVM works better with scaled data
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        logger.info(f"Train set shape: X_train={self.X_train.shape}, y_train={self.y_train.shape}")
        logger.info(f"Test set shape: X_test={self.X_test.shape}, y_test={self.y_test.shape}")
        logger.info(f"Train target distribution:\n{self.y_train.value_counts(normalize=True)}")
        logger.info(f"Test target distribution:\n{self.y_test.value_counts(normalize=True)}")
        logger.info("Applied StandardScaler to normalize features")

        return True

    def _get_optuna_objective(self):
        """
        Create and return the Optuna objective function for hyperparameter tuning.
        """

        def objective(trial):
            # Define hyperparameters to optimize
            params = {
                "C": trial.suggest_float("C", 0.01, 100.0, log=True),
                "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"]),
                "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
                "probability": True,
                "random_state": self.random_state,
            }

            # Parameters specific to non-linear kernels
            if params["kernel"] != "linear":
                params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])

            # Parameters specific to polynomial kernel
            if params["kernel"] == "poly":
                params["degree"] = trial.suggest_int("degree", 2, 5)

            model = SVC(**params)

            cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            f1_scores = []

            for fold, (train_idx, val_idx) in enumerate(cv.split(self.X_train, self.y_train)):
                X_train_fold, X_val_fold = self.X_train[train_idx], self.X_train[val_idx]
                y_train_fold, y_val_fold = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

                model.fit(X_train_fold, y_train_fold)

                preds = model.predict(X_val_fold)
                f1 = f1_score(y_val_fold, preds, average="binary")
                f1_scores.append(f1)

            return np.mean(f1_scores)

        return objective

    def tune_and_train(self):
        """
        Tune hyperparameters with Optuna and train the final model.
        """
        if self.X_train is None or self.y_train is None:
            logger.error("Training data not split. Call split_data() first.")
            return False

        logger.info(f"Running Optuna optimization with {self.n_folds} folds...")

        study = optuna.create_study(direction="maximize", study_name="svm_clinical_recurrence")
        study.optimize(self._get_optuna_objective(), n_trials=self.n_optuna_trials, show_progress_bar=True)

        self.best_params = study.best_params
        logger.info(f"Best hyperparameters found by Optuna:")
        for param, value in self.best_params.items():
            logger.info(f"  {param}: {value}")
        logger.info(f"Best CV F1-score: {study.best_value:.4f}")

        # Train final model with best parameters
        logger.info("Training final model with best parameters...")
        final_params = self.best_params.copy()
        final_params["probability"] = True
        final_params["random_state"] = self.random_state

        self.final_model = SVC(**final_params)
        self.final_model.fit(self.X_train, self.y_train)

        logger.info("Final model trained successfully.")
        return True

    def evaluate(self):
        """
        Evaluate the model on the test set.
        """
        if self.final_model is None:
            logger.error("No trained model. Call tune_and_train() first.")
            return False

        logger.info("Evaluating model on test set...")

        y_pred_test = self.final_model.predict(self.X_test)
        y_pred_proba_test = self.final_model.predict_proba(self.X_test)[:, 1]

        test_accuracy = accuracy_score(self.y_test, y_pred_test)
        test_f1 = f1_score(self.y_test, y_pred_test, average="binary")

        logger.info(f"Test Set Accuracy: {test_accuracy:.4f}")
        logger.info(f"Test Set F1 Score: {test_f1:.4f}")

        logger.info("Test Set Classification Report:")
        print(classification_report(self.y_test, y_pred_test))

        return True

    def get_feature_importance(self, top_n=20, plot=True):
        """
        Get and optionally plot feature importance.

        For SVM, feature importance is determined differently based on kernel:
        - For linear kernel: Use absolute values of coefficients
        - For non-linear kernels: Use permutation importance

        Args:
            top_n (int): Number of top features to show.
            plot (bool): Whether to generate a plot.

        Returns:
            pd.DataFrame: DataFrame of feature importances.
        """
        if self.final_model is None:
            logger.error("No trained model. Call tune_and_train() first.")
            return None

        feature_names = list(self.X.columns)

        if self.best_params["kernel"] == "linear":
            logger.info("Using coefficient values to determine feature importance (linear kernel)")
            importances = np.abs(self.final_model.coef_[0])
        else:
            logger.info("Using permutation importance to determine feature importance (non-linear kernel)")
            from sklearn.inspection import permutation_importance

            result = permutation_importance(self.final_model, self.X_test, self.y_test, n_repeats=10, random_state=self.random_state)
            importances = result.importances_mean

        importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        importance_df = importance_df.sort_values(by="Importance", ascending=False)

        logger.info(f"Top {top_n} Features:")
        print(importance_df.head(top_n))

        if plot:
            try:
                plt.figure(figsize=(10, 8))
                sns.barplot(x="Importance", y="Feature", data=importance_df.head(top_n))
                plt.title(f"Top {top_n} Feature Importances (SVM)")
                plt.tight_layout()
                plt.show()
            except Exception as e:
                logger.warning(f"Error generating plot: {e}")

        return importance_df

    def run_pipeline(self):
        """
        Run the complete pipeline from data loading to evaluation.
        """
        if not self.read_data():
            return False

        if not self.prepare_data():
            return False

        if not self.split_data():
            return False

        if not self.tune_and_train():
            return False

        if not self.evaluate():
            return False

        logger.info("Pipeline completed successfully!")
        return True


if __name__ == "__main__":
    CONFIG = {
        "CLINICAL_DATA_PATH": "data/Clinical_and_Other_Features.xlsx",
        "TARGET_COLUMN": ("Recurrence", "Recurrence event(s)", "{0 = no, 1 = yes}"),
        "TEST_SIZE": 0.20,
        "N_FOLDS": 5,
        "N_OPTUNA_TRIALS": 50,
        "RANDOM_STATE": 42,
        "TOP_N": 64,
        "FILTER_DICT": {0: ["Recurrence", "Follow Up", "US features"]},
        "EXCLUDE_COLUMNS": [("Tumor Characteristics", "Staging(Tumor Size)# [T]", ""), ("Mammography Characteristics", "Tumor Size (cm)", "")],
    }

    model = ClinicalDataSVM(
        clinical_data_path=CONFIG["CLINICAL_DATA_PATH"],
        target_column=CONFIG["TARGET_COLUMN"],
        filter_dict=CONFIG["FILTER_DICT"],
        exclude_columns=CONFIG["EXCLUDE_COLUMNS"],
        test_size=CONFIG["TEST_SIZE"],
        n_folds=CONFIG["N_FOLDS"],
        n_optuna_trials=CONFIG["N_OPTUNA_TRIALS"],
        random_state=CONFIG["RANDOM_STATE"],
    )

    model.run_pipeline()

    feature_importances = model.get_feature_importance(CONFIG["TOP_N"])
    feature_importances.to_csv("work_dirs/clinical_experiments/svm_clinic_FI.csv")
