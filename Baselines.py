# from re import VERBOSE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report,roc_auc_score, precision_recall_curve, auc, average_precision_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SMOTENC, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.datasets import fetch_datasets
from imblearn.under_sampling import EditedNearestNeighbours,TomekLinks

import seaborn as sns

import warnings
# Ignore the ConvergenceWarning from scikit-learn
warnings.filterwarnings("ignore", category=UserWarning)

def train_evaluate_model(model, X_train, X_test, y_train, y_test, verb=True):

    try:
      # Train the model
      model.fit(X_train, y_train)

      # Evaluate the model
      y_pred = model.predict(X_test)
      y_score = model.predict_proba(X_test)

      # Calculate metrics
      accuracy = accuracy_score(y_test, y_pred)
      precision = precision_score(y_test, y_pred, average='weighted')
      recall = recall_score(y_test, y_pred, average='weighted')
      f1 = f1_score(y_test, y_pred, average='weighted')

      unique_values = set(y_test)

      if len(set(y_test))==2:
        # Calculate ROC AUC score
        roc_auc = roc_auc_score(y_test, y_score[:,1], average='macro', multi_class='ovo')
        # Calculate Precision-Recall AUC score
        pr_auc = average_precision_score(y_test, y_score[:,1], average='macro')
      else:
        # Binarize the labels
        y_true_binary = preprocessing.label_binarize(y_test, classes=model.classes_)
        # Calculate ROC AUC score
        roc_auc = roc_auc_score(y_true_binary, y_score, average='macro', multi_class='ovo')
        # Calculate Precision-Recall AUC score
        pr_auc = average_precision_score(y_true_binary, y_score, average='macro')

      if verb:
          print(f"Model accuracy: {accuracy:.2f}")
          print(f"F1-score: {f1:.2f}")
          print(f"ROC AUC: {roc_auc:.3f}")
          print(f"PR AUC: {pr_auc:.3f}")

      return accuracy, precision, recall, f1, roc_auc, pr_auc

    except RuntimeError as e:
        print(f"RuntimeError occurred: {e}")
        # Handle the exception (e.g., return None or default values, or re-raise the error)
        return 0, 0, 0, 0, 0, 0

def train_evaluate_models(models, model_names,X_train, X_test, y_train, y_test,verb=True,filepath=None):
    """
    Train and evaluate multiple models and print the results.

    Parameters:
        models (list): List of models to train and evaluate.
        model_names (list): List of names corresponding to each model.
        X_test (array-like): Testing features.
        y_train (array-like): Training labels.
        y_test (array-like): Testing labels.
    """
    results=[]

    # Convert y_train to pandas Series if it's not already
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)

    if y_train.value_counts().min() < 5 or y_train.nunique() == 1:
        print("Minority class sample count is less than 5. Skipping model evaluation.")
        return 0

    for model, model_name in zip(models, model_names):
        if verb:
          print(f"{model_name} Model:")
          print(40 * '-')
        accuracy, precision, recall, f1, roc_auc, pr_auc = train_evaluate_model(model, X_train, X_test, y_train, y_test, verb)
        results.append([model_name, accuracy, precision, recall, f1, roc_auc, pr_auc])

    results_df = pd.DataFrame(results, columns=['Model', 'ACC','Precision','Recall', 'F1', 'ROC AUC', 'PR AUC'])

    if filepath is not None:
        results_df.to_csv(filepath, index=False)
    return results_df

# Function to setup models with various sampling techniques
def setup_models(X, categorical_cols, numerical_cols,simple=False):

    cat_indices = [X.columns.get_loc(col) for col in categorical_cols]

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    # Define preprocessors for numerical and categorical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', preprocessing.StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', preprocessing.OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessors into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Define the base models
    base_models = {
        "DT": DecisionTreeClassifier(random_state=1),
        "RF": RandomForestClassifier(random_state=1),
        "LR": LogisticRegression(random_state=1, max_iter=1000, solver='lbfgs'),
        "KNN": KNeighborsClassifier(),
        "LGBM": LGBMClassifier(random_state=1, verbosity=-1),
        # "LinearSVM": SVC(kernel='linear', probability=True, random_state=1)
    }

    # Define samplers
    samplers = {
        "None": None,
        "SMOTE": SMOTE(random_state=1),
        "ADASYN": ADASYN(random_state=1),
        "BorderlineSMOTE": BorderlineSMOTE(random_state=1),
        # "SMOTENC": SMOTENC(random_state=1,categorical_features=cat_indices),
        "SMOTEENN": SMOTEENN(random_state=1),
        "SMOTETomek": SMOTETomek(random_state=1)
    }

    # Create pipelines
    models = []
    model_names = []

    for model_name, model in base_models.items():
        # Add pipelines without sampling
        models.append(ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ]))
        model_names.append(f"{model_name}:Origin")

        if simple:
          continue
        else:
          # Add pipelines with samplers
          for sampler_name, sampler in samplers.items():
              if sampler is not None:
                  models.append(ImbPipeline(steps=[
                      ('preprocessor', preprocessor),
                      ('sampler', sampler),
                      ('classifier', model)
                  ]))
                  model_names.append(f"{model_name}:{sampler_name}")

    return models, model_names