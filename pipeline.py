
import numpy as np
import preprocessors as pp
import config

from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_union
from xgboost.sklearn import XGBRegressor

# %%

discrete_vars_pipeline = Pipeline([('discrete_vars_select', pp.ItemsSelector(features=config.DISCRETE_VARS))])

cat_vars_pipeline = Pipeline([('cat_vars_select', pp.ItemsSelector(features=config.CATEGORICAL_VARS)),
                              ('categorical_imputer', pp.CategoricalImputer(features=config.CATEGORICAL_VARS)),
                                ('other_label_encoder', pp.OtherLabelCategoricalEncoder(features=config.CATEGORICAL_VARS)),
                                ('categorical_encoder', pp.CategoricalEncoder(features=config.CATEGORICAL_VARS))])

numerical_vars_pipeline = Pipeline([('numerical_vars_select', pp.ItemsSelector(features=config.NUMERICAL_VARS)),
                                    ('numerical_imputer', pp.NumericalImputer(features=config.NUMERICAL_VARS))
                                    ,('power_transformer', pp.SelectedFeaturesPowerTransformer(features=config.NUMERICAL_VARS))
                                    ])

temporal_vars_pipeline = Pipeline([('temporal_vars_select', pp.ItemsSelector(features=config.TEMPORAL_VARS)),
                                   ('temporal_processor', pp.TemporalVariableTransformer(features=config.TEMPORAL_VARS,
                                                                                       reference_feature=config.TEMPORAL_REFERENCE_VAR))])

vars_pipeline = make_union(discrete_vars_pipeline, cat_vars_pipeline, numerical_vars_pipeline, temporal_vars_pipeline)

total_pipeline = Pipeline([('vars_preprocessing', vars_pipeline)
                            ,('scale', StandardScaler())
                           ,('ml_model', TransformedTargetRegressor(XGBRegressor(learning_rate=0.05, n_estimators=400, random_state=1),
                                                                   func = np.log, inverse_func=np.exp))])





# %%

