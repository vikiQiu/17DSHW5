__author__ = "Victoria"

# 2017-12-1

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from scipy.special import boxcox1p
from scipy.stats import norm, skew

class Process:
    def __init__(self, coly='SalePrice'):
        self.coly = coly

        train = self.read_data('../data/trainData.csv')
        # outliers
        train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)
        test = self.read_data('../data/testData.csv')
        df = self.concat_df(train, test)
        self.train, self.test = self.part_df(self.preprocess(df))

        self.trainy = self.train[coly]
        self.trainx = self.train.loc[:, [c != coly for c in self.train.columns]]

    def read_data(self, filename):
        df = pd.read_csv(filename, na_filter=False)
        df = df.set_index('Id')
        return df

    def preprocess(self, df):
        '''
        Preprocess function.
        :param df: The DataFrame with trian and test data.
        :return:
        '''
        # cols = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'FullBath', 'TotRmsAbvGrd',
        #         'YearBuilt', 'SalePrice', 'train', 'Electrical']
        # df = df[cols]
        df[self.coly] = np.log(df[self.coly])
        df = self.deal_with_nan(df)
        # df['GrLivArea2'] = np.log(df['GrLivArea'])
        df['HasBsmt'] = pd.Series(len(df['TotalBsmtSF']), index=df.index)
        df['HasBsmt'] = 0
        # df.loc[df['TotalBsmtSF'] > 0, 'HasBsmt'] = 1
        # df.loc[df['TotalBsmtSF'] == 0, 'TotalBsmtSF'] = 1
        # df['TotalBsmtSF2'] = np.log(df['TotalBsmtSF'])

        # Some feature are category
        # MSSubClass=The building class
        df['MSSubClass'] = df['MSSubClass'].apply(str)
        # Changing OverallCond into a categorical variable
        df['OverallCond'] = df['OverallCond'].astype(str)
        # Year and month sold are transformed into categorical features.
        df['YrSold'] = df['YrSold'].astype(str)
        df['MoSold'] = df['MoSold'].astype(str)
        df['HasBsmt'] = df['HasBsmt'].astype(str)

        # df = self.numeric_transform(df)

        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

        # cols = ['YearBuilt']
        # df = pd.get_dummies(df, columns=cols)
        return df

    def deal_with_nan(self, df):
        '''
        Deal with NaN of column to column in df
        :param df: A DataFrame
        :return:
        '''
        # NaN means None in description
        cols_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                     'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                     'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                     'MasVnrType', 'Functional', 'Electrical', 'KitchenQual', 'Exterior1st',
                     'Exterior2nd', 'SaleType', 'MSSubClass']
        for col in cols_none:
            df[col].fillna('None', inplace=True)

        # NaN fills with 0
        cols_zero = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2',
                     'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea', 'MSZoning']
        for col in cols_zero:
            df[col].fillna(0, inplace=True)

        df = df.drop(['Utilities'], axis=1)

        cols = df.columns
        n = len(df)
        for i, col in enumerate(cols):
            # print('Processing [%d] column %s' % (i, col))
            if col == self.coly:
                continue
            if df[col].dtypes == 'O':
                ind = [type(x) == float for x in df[col]]
            else:
                ind = [np.isnan(x) for x in df[col]]
            if sum(ind) / n > 0.0001:
                # TODO: A better way to deal with object column's NaN problem
                del df[col]
            else:
                if df[col].dtypes == 'O':
                    del df[col]
                else:
                    df[col].fillna(df[col].median(), inplace=True)
        return df

    def numeric_transform(self, df):
        numeric_feats = [col for col in df.columns if col != self.coly and df[col].dtypes != 'O' and col != 'train']

        # Check the skew of all numerical features
        skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        print("\nSkew in numerical features: \n")
        skewness = pd.DataFrame({'Skew': skewed_feats})
        skewness.head(10)

        skewness = skewness[abs(skewness) > 0.75]
        print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

        skewed_features = skewness.index
        lam = 0.15
        for feat in skewed_features:
            # all_data[feat] += 1
            df[feat] = boxcox1p(df[feat], lam)

            # df[skewed_features] = np.log1p(df[skewed_features])
        return df

    def concat_df(self, train, test):
        '''
        Concat train and test data with label
        :param train:
        :param test:
        :return:
        '''
        train = train.assign(train=1)
        test = test.assign(train=0)
        df = pd.concat([train, test])
        return df

    def part_df(self, df: pd.DataFrame):
        '''
        Part train and test data from df.
        :param df: A DataFrame with 'train' feature
        :return:
        '''
        train = df[df.train == 1]
        del train['train']
        train = train.reset_index(drop=True)
        test = df[df.train == 0]
        del test['train']
        del test[self.coly]
        return train, test

    def test_model(self, K, mols, epoch=1):
        '''
        Train the model
        :param K: K-fold Cross Validation.
        :param mol: The model to test.
        :return: A model
        '''
        kf = KFold(n_splits=K, shuffle=True)
        mses = []
        for train_index, test_index in kf.split(self.trainx):
            mse = []
            for i in range(epoch):
                pres = []
                for mol in mols:
                    mol.fit(self.trainx.iloc[train_index], self.trainy.iloc[train_index])
                    pres.append(mol.predict(self.trainx.iloc[test_index]))
                pre = np.mean(np.array(pres), axis=0)
                mse.append(self.get_mse(pre, self.trainy.iloc[test_index]))
            print('Prediction std=%f, MSE=%f' % (np.std(pre), np.mean(mse)))
            mses.append(np.mean(mse))
        mse = np.mean(mses)
        print('Prediction std=%f, MSE=%f' % (np.std(pre), mse))
        print('[Fin]%d fold Prediction MSE=%f' % (K, mse))

        return mse

    def get_mse(self, pre, y):
        # return np.sqrt(np.mean((pre - y) ** 2))
        return np.sqrt(np.mean((np.exp(pre) - np.exp(y)) ** 2))

    def get_pre(self, mols, filename='sub1.csv', dirname='../submission/'):
        pre = []
        for mol in mols:
            mol.fit(self.trainx, self.trainy)
            # pre.append(mol.predict(self.test))
            pre.append(np.exp(mol.predict(self.test)))
        pre = np.mean(np.array(pre), axis=0)
        res = pd.DataFrame({'Id': self.test.index, self.coly: pre})
        res.to_csv(dirname + filename, index=False)
        return pre


if __name__ == '__main__':
    process = Process()
    lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
    ridge = Ridge(alpha=1)
    lr = LinearRegression()
    rf = RandomForestRegressor()
    gbd = GradientBoostingRegressor(n_estimators=100)
    k_ridge = KernelRidge(alpha=0.6, kernel='polynomial', degree=1, coef0=2.5)
    e_net = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
    process.test_model(5, [lasso])
    process.test_model(5, [ridge])
    process.test_model(5, [lr])
    process.test_model(5, [rf])
    process.test_model(5, [gbd])
    process.test_model(5, [k_ridge])
    process.test_model(5, [e_net])
    process.test_model(5, [gbd, e_net, lr])
    # process.test_model(5, make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)))
    # process.get_pre([Ridge(alpha=1.0)])
    # process.get_pre([RandomForestRegressor()], filename='sub2.csv')
    # process.get_pre([gbd, e_net, lr], filename='sub3.csv')
    # print(process.train.dtypes)
