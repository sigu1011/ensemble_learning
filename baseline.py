import re
import numpy as np
import pandas as pd
import support
from sklearn.model_selection import KFold, cross_validate
from sklearn.svm import SVC, SVR
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

""" Training Dataset as:
https://archive.ics.uci.edu/ml/datasets/Iris/
https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+%28Sonar%2C+Mines+vs.+Rocks%29
https://archive.ics.uci.edu/ml/machine-learning-databases/glass/
https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise
https://archive.ics.uci.edu/ml/datasets/Wine+Quality
"""

if __name__ == '__main__':
    # ベンチマークとなるアルゴリズムと、アルゴリズムを実装したモデルの一覧
    models = [
        ('SVM', SVC(gamma='auto', random_state=1), SVR(gamma='auto')),
        ('GaussianProcess', GaussianProcessClassifier(random_state=1),
         GaussianProcessRegressor(normalize_y=True, alpha=1, random_state=1)),
        ('KNeighbors', KNeighborsClassifier(), KNeighborsRegressor()),
        ('MLP', MLPClassifier(hidden_layer_sizes=(100,), random_state=1),
         MLPRegressor(hidden_layer_sizes=(5,), solver='lbfgs', random_state=1)),
    ]

    # 検証用データセットのファイルと、ファイルの区切り文字、
    # ヘッダーとなる行の位置、インデックスとなる列の位置のリスト
    classifier_files = ['iris.data', 'sonar.all-data', 'glass.data']
    classifier_params = [(',', None, None), (',', None, None), (',', None, 0)]
    regressor_files = ['airfoil_self_noise.dat', 'winequality-red.csv', 'winequality-white.csv']
    regressor_params = [(r'\t', None, None), (';', 0, None), (';', 0, None)]

    # 評価スコアを、検証用データセットのファイル、アルゴリズム毎に保存する表
    result = pd.DataFrame(columns=['dataset', 'function'] + [model[0] for model in models],
                          index=range(len(classifier_files + regressor_files) * 2))

    # クラス分類アルゴリズムの評価
    column_num = 0
    for i, (file, param) in enumerate(zip(classifier_files, classifier_params)):
        # ファイルの読み込み
        df = pd.read_csv(file, sep=param[0], header=param[1], index_col=param[2], engine='python')
        train = df[df.columns[:-1]].values
        target, clz = support.clz_to_prob(df[df.columns[-1]])

        # 結果の表を事前に作成する
        # ファイル名からデータセットの種類と、評価関数用の行を作る
        result.loc[column_num, 'dataset'] = re.split(r'[._]', file)[0]
        result.loc[column_num + 1, 'dataset'] = ''
        result.loc[column_num, 'function'] = 'F1Score'
        result.loc[column_num + 1, 'function'] = 'Accuracy'

        # 全アルゴリズムで評価を行う
        for model_name, classifier, _ in models:
            # sklearnの関数で交差検証した結果のスコアを取得
            kf = KFold(n_splits=5, random_state=1, shuffle=True)
            score = cross_validate(classifier, train, target.argmax(axis=1), cv=kf, scoring=('f1_weighted', 'accuracy'))
            result.loc[column_num, model_name] = np.mean(score['test_f1_weighted'])
            result.loc[column_num + 1, model_name] = np.mean(score['test_accuracy'])

        column_num += 2

    # 回帰アルゴリズムの評価
    for i, (file, param) in enumerate(zip(regressor_files, regressor_params)):
        # ファイルの読み込み
        df = pd.read_csv(file, sep=param[0], header=param[1], index_col=param[2], engine='python')
        train = df[df.columns[:-1]].values
        target = df[df.columns[-1]].values.reshape((-1,))

        # 結果の表を事前に作成する
        # ファイル名からデータセットの種類と、評価関数用の行を作る
        result.loc[column_num, 'dataset'] = re.split(r'[._]', file)[0]
        result.loc[column_num + 1, 'dataset'] = ''
        result.loc[column_num, 'function'] = 'R2Score'
        result.loc[column_num + 1, 'function'] = 'MeanSquared'

        # 全アルゴリズムで評価を行う
        for model_name, _, regressor in models:
            # sklearnの関数で交差検証した結果のスコアを取得
            kf = KFold(n_splits=5, random_state=1, shuffle=True)
            score = cross_validate(regressor, train, target, cv=kf, scoring=('r2', 'neg_mean_squared_error'))
            result.loc[column_num, model_name] = np.mean(score['test_r2'])
            # 符号を反転させ、もとの二条平均誤差を取得
            result.loc[column_num + 1, model_name] = -np.mean(score['test_neg_mean_squared_error'])

        column_num += 2

    pd.set_option('display.width', 400)
    pd.set_option('display.max_columns', 10)
    print(result)
    result.to_csv('baseline.csv', index=None)