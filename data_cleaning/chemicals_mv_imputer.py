from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Lasso


def impute_abbrev_mv(abbrev):
    non_chemical_columns = ["ndb", "Shrt_Desc", "Refuse_Pct", "GmWt_1", "GmWt_Desc1", "GmWt_2", "GmWt_Desc2"]

    chemicals = abbrev.drop(columns=non_chemical_columns)
    for column in chemicals.columns:
        chemicals[column] = chemicals[column].apply(lambda x: x if type(x) != str else float(x.replace(",", ".")))

    columns_for_linear_imputation, columns_for_mean_imputation = get_split_columns(chemicals)

    abbrev[columns_for_linear_imputation] = impute(chemicals[columns_for_linear_imputation],
                                                   IterativeImputer(estimator=Lasso(), max_iter=20))

    for column in columns_for_linear_imputation:
        top_bound, mean = abbrev[column].quantile(.95), abbrev[column].mean()
        abbrev[column] = abbrev[column].apply(lambda x: 0 if x < 0 else (x if x < top_bound else mean))

    abbrev[columns_for_mean_imputation] = impute(chemicals[columns_for_mean_imputation], SimpleImputer(strategy="mean"))


def get_split_columns(chemicals):
    columns_for_linear_imputation, columns_for_mean_imputation = list(chemicals.columns), []
    corr = chemicals.corr()
    threshold = .5

    for column in chemicals.columns:
        if corr[column].apply(abs).nlargest(2)[1] < threshold:
            columns_for_linear_imputation.remove(column)
            columns_for_mean_imputation.append(column)

    return columns_for_linear_imputation, columns_for_mean_imputation


def impute(df, imp):
    return imp.fit_transform(df)
