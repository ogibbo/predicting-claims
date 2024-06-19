from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

from scripts.tp_col_groups import TP_INJURY_COLS, TP_TYPE_COLS

def create_preprocessing_pipeline(num_cols, ohe_cols):
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    ohe_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    col_trans = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_cols),
        ('ohe', ohe_pipeline, ohe_cols),
    ])

    pipeline = Pipeline(steps=[
    ('preprocessing', col_trans)
    ])

    return pipeline.set_output(transform='pandas')