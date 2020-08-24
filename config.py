# %%

TARGET = 'SalePrice'

# input variables
FEATURES = ['OverallQual', 'OverallCond', 'GarageCars', 'MSZoning', 'Neighborhood', 'BsmtExposure', 'CentralAir',
            'KitchenQual', 'FireplaceQu', 'GarageType', 'SaleCondition', 'LotArea', 'BsmtFinSF1', '1stFlrSF',
            'GrLivArea', 'GrLivAndBsmtArea', 'YearBuilt', 'YearRemodAdd', 'YrSold']

# this variable is to calculate the temporal variable,
# must be dropped afterwards

DISCRETE_VARS = ['OverallQual', 'OverallCond', 'GarageCars']

TEMPORAL_VARS = ['YearBuilt', 'YearRemodAdd', 'YrSold']

TEMPORAL_REFERENCE_VAR = 'YrSold'

GARAGEYRBLT_VAR = 'GarageYrBlt'

# variables to power transform
NUMERICAL_VARS = ['LotArea', 'BsmtFinSF1', '1stFlrSF', 'GrLivArea', 'GrLivAndBsmtArea']

# categorical variables to encode
CATEGORICAL_VARS = ['MSZoning', 'Neighborhood', 'BsmtExposure', 'CentralAir', 'KitchenQual', 'FireplaceQu', 'GarageType', 'SaleCondition']

