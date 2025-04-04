import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None


# Define PyTorch MLP class outside the main class
class MLP(nn.Module):
    """PyTorch Multi-Layer Perceptron with configurable architecture"""

    def __init__(self, n_features, hidden_sizes, dropout_rate, activation_fn, n_classes=2):
        super(MLP, self).__init__()

        if activation_fn == "relu":
            self.activation = nn.ReLU()
        elif activation_fn == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Sigmoid()

        layers = [nn.Linear(n_features, hidden_sizes[0]), self.activation]

        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(self.activation)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(hidden_sizes[-1], n_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def predict(self, X):
        """Get class predictions from probability outputs."""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(next(self.parameters()).device)
            outputs = self(X_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.cpu().numpy()

    def predict_proba(self, X):
        """Get probability predictions."""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(next(self.parameters()).device)
            outputs = self(X_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            return probs


class ClinicalDataMLP:
    """A class to handle loading, processing, and modeling clinical data using MLP with CUDA support."""

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
        use_cuda=True,
    ):
        """
        Initialize the model with configuration parameters.

        Args:
            clinical_data_path (str): Path to the Excel file with clinical data.
            target_column (tuple): MultiIndex column tuple for the target variable.
            filter_dict (dict, optional): Dictionary with level indices as keys and lists of values to filter out.
            exclude_columns (list, optional): List of specific column tuples to exclude.
            test_size (float): Proportion of data to use for testing.
            n_folds (int): Number of cross-validation folds.
            n_optuna_trials (int): Number of Optuna trials for hyperparameter search.
            random_state (int): Random seed for reproducibility.
            use_cuda (bool): Whether to use CUDA GPU acceleration if available.
        """
        self.clinical_data_path = clinical_data_path
        self.target_column = target_column
        self.filter_dict = filter_dict
        self.exclude_columns = exclude_columns
        self.test_size = test_size
        self.n_folds = n_folds
        self.n_optuna_trials = n_optuna_trials
        self.random_state = random_state
        self.use_cuda = use_cuda and torch.cuda.is_available()

        if self.use_cuda:
            logger.info("CUDA is available. GPU acceleration will be used.")
        else:
            logger.info("CUDA is not available or disabled. Using CPU only.")

        # Set the device
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

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
        self.feature_names = None

        # Set random seeds for reproducibility
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if self.use_cuda:
            torch.cuda.manual_seed(self.random_state)

    def read_data(self):
        """
        Load clinical data from an Excel file with a multi-index header.
        """
        try:
            # Load data with multi-index header
            self.df = pd.read_excel(self.clinical_data_path, header=[0, 1, 2])

            # Clean up unnamed columns in multi-index
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

        # Flatten multi-index column names
        flat_X_columns = ["_".join(filter(None, map(str, col))).strip("_") for col in original_X_columns]
        self.X.columns = flat_X_columns

        self._process_features(df_processed, original_X_columns)

        # Store feature names before conversion to tensors
        self.feature_names = self.X.columns

        return True

    def filter_features(self):
        """
        Filter features based on multi-level headers.

        Returns:
            pd.DataFrame: Filtered DataFrame with specified columns removed.
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
                    # Filter columns where the level value matches any value in values_to_filter
                    for value in values_to_filter:
                        level_matches = all_columns[all_columns.get_level_values(level) == value]
                        logger.info(f"Found {len(level_matches)} columns with '{value}' at level {level}")
                        columns_to_drop.extend(level_matches.tolist())
            else:
                # Fallback for non-MultiIndex columns
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
            # Show a sample of columns to be dropped
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
        # Identify numerical and categorical columns
        numerical_cols = []
        categorical_cols = []

        col_mapping_flat_to_original = {flat: orig for flat, orig in zip(self.X.columns, original_X_columns)}

        for i, col in enumerate(self.X.columns):
            original_col_tuple = original_X_columns[i]
            # Try to convert to numeric, setting errors to 'coerce'
            self.X[col] = pd.to_numeric(self.X[col], errors="coerce")

            # Use original column data to determine type
            original_col_data_series = df_processed[original_col_tuple]

            if pd.api.types.is_numeric_dtype(self.X[col]) and not self.X[col].isnull().all():
                numerical_cols.append(col)
            else:
                categorical_cols.append(col)
                self.X[col] = original_col_data_series.astype(str)

        logger.info(f"Identified {len(numerical_cols)} numerical columns.")
        logger.info(f"Identified {len(categorical_cols)} categorical columns.")

        # Handle missing values in numerical columns
        if numerical_cols:
            num_imputer = SimpleImputer(strategy="median")
            self.X[numerical_cols] = num_imputer.fit_transform(self.X[numerical_cols])
            logger.info("Imputed missing values in numerical columns using median.")

        # Handle categorical columns
        if categorical_cols:
            # Impute missing values in categorical columns
            cat_imputer = SimpleImputer(strategy="constant", fill_value="Missing")
            self.X[categorical_cols] = cat_imputer.fit_transform(self.X[categorical_cols])
            logger.info("Imputed missing values in categorical columns with 'Missing'.")

            # One-hot encode categorical columns
            self.X = pd.get_dummies(self.X, columns=categorical_cols, drop_first=True, dummy_na=False, dtype=int)
            logger.info("Applied one-hot encoding to categorical columns.")
            logger.info(f"Data shape after encoding: {self.X.shape}")

            # Clean column names for compatibility
            logger.info("Cleaning column names for MLP compatibility...")
            original_cols = self.X.columns.tolist()
            self.X.columns = self.X.columns.str.replace("[\[\]<]", "_", regex=True)
            cleaned_cols = self.X.columns.tolist()

            changed_cols = [(orig, clean) for orig, clean in zip(original_cols, cleaned_cols) if orig != clean]
            if changed_cols:
                logger.info(f"Cleaned {len(changed_cols)} column names.")

        # Final check for non-numeric columns
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

        # Scale the features - essential for neural networks
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        logger.info(f"Train set shape: X_train={self.X_train.shape}, y_train={self.y_train.shape}")
        logger.info(f"Test set shape: X_test={self.X_test.shape}, y_test={self.y_test.shape}")
        logger.info(f"Train target distribution:\n{self.y_train.value_counts(normalize=True)}")
        logger.info(f"Test target distribution:\n{self.y_test.value_counts(normalize=True)}")

        return True

    def _get_optuna_objective(self):
        """
        Create and return the Optuna objective function for hyperparameter tuning with PyTorch.
        """

        def objective(trial):
            # PyTorch MLP parameter space
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
            dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
            activation_fn = trial.suggest_categorical("activation", ["relu", "tanh", "sigmoid"])
            optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd"])

            # Define hidden layers
            n_layers = trial.suggest_int("n_layers", 1, 3)
            n_units = []
            for i in range(n_layers):
                n_units.append(trial.suggest_int(f"n_units_l{i}", 32, 512))

            # Define input and output dimensions
            n_features = self.X_train.shape[1]

            # Set up cross-validation
            cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            f1_scores = []

            for fold, (train_idx, val_idx) in enumerate(cv.split(self.X_train, self.y_train)):
                # Convert indices to actual data
                X_train_fold = self.X_train[train_idx]
                X_val_fold = self.X_train[val_idx]
                y_train_fold = self.y_train.iloc[train_idx].values
                y_val_fold = self.y_train.iloc[val_idx].values

                # Convert to PyTorch tensors
                X_train_tensor = torch.FloatTensor(X_train_fold).to(self.device)
                y_train_tensor = torch.LongTensor(y_train_fold).to(self.device)
                X_val_tensor = torch.FloatTensor(X_val_fold).to(self.device)
                y_val_tensor = torch.LongTensor(y_val_fold).to(self.device)

                # Create data loaders
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                # Create model
                model = MLP(n_features, n_units, dropout_rate, activation_fn).to(self.device)

                # Define loss function
                criterion = nn.CrossEntropyLoss()

                # Define optimizer
                if optimizer_name == "adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                else:  # sgd
                    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

                # Train the model
                model.train()
                epochs = 100  # Maximum epochs
                patience = 10  # Early stopping patience
                best_val_loss = float("inf")
                no_improve_epochs = 0

                for epoch in range(epochs):
                    for X_batch, y_batch in train_loader:
                        # Forward pass
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)

                        # Backward and optimize
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # Evaluate on validation set
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_val_tensor)
                        val_loss = criterion(val_outputs, y_val_tensor)
                        val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()

                    # Check early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        no_improve_epochs = 0
                    else:
                        no_improve_epochs += 1
                        if no_improve_epochs >= patience:
                            break

                    model.train()

                # Final evaluation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()

                f1 = f1_score(y_val_fold, val_preds, average="binary")
                f1_scores.append(f1)

            # Return the mean F1 score across all folds
            return np.mean(f1_scores)

        return objective

    def tune_and_train(self):
        """
        Tune hyperparameters with Optuna and train the final PyTorch model with CUDA.
        """
        if self.X_train is None or self.y_train is None:
            logger.error("Training data not split. Call split_data() first.")
            return False

        logger.info(f"Running Optuna optimization with {self.n_optuna_trials} trials...")

        study = optuna.create_study(direction="maximize", study_name="mlp_clinical_recurrence")
        study.optimize(self._get_optuna_objective(), n_trials=self.n_optuna_trials, show_progress_bar=True)

        self.best_params = study.best_params
        logger.info(f"Best hyperparameters found by Optuna:")
        for param, value in self.best_params.items():
            logger.info(f"  {param}: {value}")
        logger.info(f"Best CV F1-score: {study.best_value:.4f}")

        logger.info("Training final model with best parameters...")

        # Extract model architecture parameters
        n_features = self.X_train.shape[1]
        n_layers = self.best_params["n_layers"]
        hidden_sizes = [self.best_params[f"n_units_l{i}"] for i in range(n_layers)]
        dropout_rate = self.best_params["dropout_rate"]
        activation_fn = self.best_params["activation"]
        learning_rate = self.best_params["learning_rate"]
        weight_decay = self.best_params["weight_decay"]
        optimizer_name = self.best_params["optimizer"]
        batch_size = self.best_params["batch_size"]

        # Create the final model
        self.final_model = MLP(n_features, hidden_sizes, dropout_rate, activation_fn).to(self.device)

        # Define loss function
        criterion = nn.CrossEntropyLoss()

        # Define optimizer
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.final_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:  # sgd
            optimizer = torch.optim.SGD(self.final_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

        # Split training data to get a validation set for early stopping
        X_train_final, X_eval_final, y_train_final, y_eval_final = train_test_split(
            self.X_train, self.y_train, test_size=0.1, random_state=self.random_state, stratify=self.y_train
        )

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_final).to(self.device)
        y_train_tensor = torch.LongTensor(y_train_final.values).to(self.device)
        X_eval_tensor = torch.FloatTensor(X_eval_final).to(self.device)
        y_eval_tensor = torch.LongTensor(y_eval_final.values).to(self.device)

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Train the model
        self.final_model.train()
        epochs = 200  # Maximum epochs
        patience = 20  # Early stopping patience
        best_val_loss = float("inf")
        no_improve_epochs = 0

        logger.info("Training final model...")
        for epoch in range(epochs):
            # Training loop
            self.final_model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                # Forward pass
                outputs = self.final_model(X_batch)
                loss = criterion(outputs, y_batch)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * X_batch.size(0)

            train_loss = train_loss / len(train_dataset)

            # Evaluate on validation set
            self.final_model.eval()
            with torch.no_grad():
                val_outputs = self.final_model(X_eval_tensor)
                val_loss = criterion(val_outputs, y_eval_tensor).item()
                _, val_preds = torch.max(val_outputs, 1)
                val_accuracy = (val_preds == y_eval_tensor).sum().item() / len(y_eval_tensor)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

            # Check early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_epochs = 0
                # Save best model weights
                best_model_weights = self.final_model.state_dict().copy()
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model weights
        self.final_model.load_state_dict(best_model_weights)

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

        # Make predictions with the PyTorch model
        self.final_model.eval()
        y_pred_test = self.final_model.predict(self.X_test)
        y_pred_proba_test = self.final_model.predict_proba(self.X_test)[:, 1]

        test_accuracy = accuracy_score(self.y_test, y_pred_test)
        test_f1 = f1_score(self.y_test, y_pred_test, average="binary")

        logger.info(f"Test Set Accuracy: {test_accuracy:.4f}")
        logger.info(f"Test Set F1 Score: {test_f1:.4f}")

        logger.info("Test Set Classification Report:")
        print(classification_report(self.y_test, y_pred_test))

        # Create confusion matrix
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(pd.crosstab(self.y_test, pd.Series(y_pred_test, name="Predicted"), normalize="index"), annot=True, fmt=".2%", cmap="Blues")
            plt.title("Confusion Matrix")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.warning(f"Error generating confusion matrix: {e}")

        return True

    def get_feature_importance(self, top_n=20, plot=True):
        """
        Get and optionally plot feature importance for PyTorch MLP.

        This is an approximation since neural networks don't provide direct feature importance.
        We use a permutation approach to estimate feature importance.

        Args:
            top_n (int): Number of top features to show.
            plot (bool): Whether to generate a plot.

        Returns:
            pd.DataFrame: DataFrame of feature importances.
        """
        if self.final_model is None:
            logger.error("No trained model. Call tune_and_train() first.")
            return None

        logger.info("Calculating permutation feature importance (this may take a while)...")

        # Use permutation importance
        # For each feature, shuffle its values and see how much the performance drops
        self.final_model.eval()
        baseline_pred = self.final_model.predict(self.X_test)
        baseline_score = f1_score(self.y_test, baseline_pred, average="binary")

        importance = []

        for col_idx in tqdm(range(self.X_test.shape[1]), desc="Processed features"):
            # Make a copy of the test data
            X_test_permuted = self.X_test.copy()

            # Shuffle one feature
            np.random.seed(self.random_state)
            X_test_permuted[:, col_idx] = np.random.permutation(X_test_permuted[:, col_idx])

            # Predict with shuffled feature
            preds_permuted = self.final_model.predict(X_test_permuted)
            score_permuted = f1_score(self.y_test, preds_permuted, average="binary")

            # The importance is the drop in performance
            feature_importance = baseline_score - score_permuted
            importance.append(feature_importance)

        # Create DataFrame with feature importances
        importance_df = pd.DataFrame({"Feature": self.feature_names, "Importance": importance})

        # Sort by importance
        importance_df = importance_df.sort_values(by="Importance", ascending=False)

        logger.info(f"Top {top_n} Features (by permutation importance):")
        print(importance_df.head(top_n))

        if plot:
            try:
                plt.figure(figsize=(10, 8))
                sns.barplot(x="Importance", y="Feature", data=importance_df.head(top_n))
                plt.title(f"Top {top_n} Feature Importances (MLP - Permutation Method)")
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

    model = ClinicalDataMLP(
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
    feature_importances.to_csv("work_dirs/clinical_experiments/mlp_clinic_FI.csv")
