import numpy as np
import support


class ZeroRule:
    def __init__(self):
        self.r = None

    def fit(self, x, y):
        self.r = np.mean(y, axis=0)
        return self

    def predict(self, x):
        z = np.zeros((len(x), self.r.shape[0]))
        return z + self.r

    def __str__(self):
        return str(self.r)


if __name__ == '__main__':
    import pandas as pd

    ps = support.get_base_args()
    args = ps.parse_args()
    df = pd.read_csv(args.input, sep=args.separator, header=args.header, index_col=args.indexcol, engine='python')
    x = df[df.columns[:-1]].values

    if not args.regression:
        y, clz = support.clz_to_prob(df[df.columns[-1]])
        plf = ZeroRule()
        support.report_classifier(plf, x, y, clz, args.crossvalidate)
    else:
        y = df[df.columns[-1]].values.reshape((-1, 1))
        plf = ZeroRule()
        support.report_regressor(plf, x, y, args.crossvalidate)
