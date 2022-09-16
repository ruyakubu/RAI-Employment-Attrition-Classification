
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def split_label(dataset, target_feature):
    X = dataset.drop([target_feature], axis=1)
    y = dataset[[target_feature]]
    return X, y

def transform_data(X, y, target_feature):
    features = X.columns.values.tolist()
    classes = y[target_feature].unique().tolist()
    pipe_cfg = {
        'num_cols': X.dtypes[X.dtypes == 'int64'].index.values.tolist(),
        'cat_cols': X.dtypes[X.dtypes == 'object'].index.values.tolist(),
    }
    num_pipe = Pipeline([
        ('num_imputer', SimpleImputer(strategy='median')),
        ('num_scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('cat_imputer', SimpleImputer(strategy='constant', fill_value='?')),
        ('cat_encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    feat_pipe = ColumnTransformer([
        ('num_pipe', num_pipe, pipe_cfg['num_cols']),
        ('cat_pipe', cat_pipe, pipe_cfg['cat_cols'])
    ])
    X = feat_pipe.fit_transform(X)
    print(pipe_cfg['cat_cols'])
    return X, feat_pipe, features, classes

def get_categorical_numerical_data(all_data):
    categorical = []
    for col, value in all_data.iteritems():
        if value.dtype == 'object':
            categorical.append(col)
    numerical = all_data.columns.difference(categorical)
    #categorical.drop('Attrition_numerical')
    categorical.remove('Attrition_numerical')
    return categorical, numerical
