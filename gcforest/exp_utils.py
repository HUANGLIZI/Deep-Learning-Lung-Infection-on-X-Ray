"""
Description: A python 2.7 implementation of gcForest proposed in [1]. A demo implementation of gcForest library as well as some demo client scripts to demostrate how to use the code. The implementation is flexible enough for modifying the model or
fit your own datasets. 
Reference: [1] Z.-H. Zhou and J. Feng. Deep Forest: Towards an Alternative to Deep Neural Networks. In IJCAI-2017.  (https://arxiv.org/abs/1702.08835v2 )
Requirements: This package is developed with Python 2.7, please make sure all the demendencies are installed, which is specified in requirements.txt
ATTN: This package is free for academic usage. You can run it at your own risk. For other purposes, please contact Prof. Zhi-Hua Zhou(zhouzh@lamda.nju.edu.cn)
ATTN2: This package was developed by Mr.Ji Feng(fengj@lamda.nju.edu.cn). The readme file and demo roughly explains how to use the codes. For any problem concerning the codes, please feel free to contact Mr.Feng. 
"""
import numpy as np
from scipy.sparse import issparse

from .utils.log_utils import get_logger

LOGGER = get_logger('gcforest.exp_utils')


# 从json文件中加载配置信息
def load_model_config(model_path, log_name=None):
    import json
    from .utils.config_utils import load_json
    config = load_json(model_path)
    if log_name is not None:
        logger = get_logger(log_name)
        logger.info(log_name)
        logger.info("\n" + json.dumps(config, sort_keys=True, indent=4, separators=(',', ':')))   # 将python对象转成json字符串
    return config


# 多维转2维，数组拼接
def concat_datas(datas):
    if type(datas) != list:
        return datas
    for i, data in enumerate(datas):
        datas[i] = data.reshape((data.shape[0], -1))  # 多维转换成2维
    return np.concatenate(datas, axis=1)  # 在行上拼接数组，行数增加


# 归一化 每列代表一个特征，要将每个特征的取值范围限制在同样的范围内，所以是对列做归一化，方便梯度下降。
def data_norm(X_train, X_test):
    X_mean = np.mean(X_train, axis=0)   # 对列求均值
    X_std = np.std(X_train, axis=0)    # 对列求标准差
    X_train -= X_mean
    X_train /= X_std

    # 对测试集做跟训练集同样的变化，不单独对测试集做归一化是因为测试集不需要拿去训练，
    # 所以只要保持相同的变换即可。
    X_test -= X_mean
    X_test /= X_std
    return X_mean, X_std


# 对应行拼接
def append_origin(X, X_origin):
    return np.hstack((X.reshape((X.shape[0]), -1), X_origin.reshape((X_origin.shape[0], -1))))


# ET
def prec_ets(n_trees, X_train, y_train, X_test, y_test, random_state=None):
    """
    ExtraTrees
    """
    from sklearn.ensemble import ExtraTreesClassifier    #极端森林分类器
    if not issparse(X_train):    # 不是以稀疏矩阵方式存储
        X_train = X_train.reshape((X_train.shape[0], -1))   #转换成二维矩阵
    if not issparse(X_test):
        X_test = X_test.reshape((X_test.shape[0], -1))
    LOGGER.info('start predict: n_trees={},X_train.shape={},y_train.shape={},X_test.shape={},y_test.shape={}'.format(
        n_trees, X_train.shape, y_train.shape, X_test.shape, y_test.shape))
    # n_job: 并行作业数
    clf = ExtraTreesClassifier(n_estimators=n_trees, max_depth=None, n_jobs=-1, verbose=1, random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    prec = float(np.sum(y_pred == y_test)) / len(y_test)   # 预测准确度
    LOGGER.info('prec_ets{}={:.6f}%'.format(n_trees, prec*100.0))
    return clf, y_pred

# RF
def prec_rf(n_trees, X_train, y_train, X_test, y_test):
    """
    RandomForest
    """
    from sklearn.ensemble import RandomForestClassifier   # 随机森林分类器
    if not issparse(X_train):
        X_train = X_train.reshape((X_train.shape[0], -1))
    if not issparse(X_test):
        X_test = X_test.reshape((X_test.shape[0], -1))
    LOGGER.info('start predict: n_trees={},X_train.shape={},y_train.shape={},X_test.shape={},y_test.shape={}'.format(
        n_trees, X_train.shape, y_train.shape, X_test.shape, y_test.shape))
    # n_jobs=-1 表示使用所有处理器
    clf = RandomForestClassifier(n_estimators=n_trees, max_depth=None, n_jobs=-1, verbose=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    prec = float(np.sum(y_pred == y_test)) / len(y_test)
    LOGGER.info('prec_rf{}={:.6f}%'.format(n_trees, prec*100.0))
    return clf, y_pred


# ？
def xgb_eval_accuracy(y_pred_proba, y_true):
    """
    y_true (DMatrix)
    """
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = y_true.get_label()
    acc = float(np.sum(y_pred == y_true)) / len(y_pred)
    return 'accuracy', -acc   # ？

def prec_xgb(n_trees, max_depth, X_train, y_train, X_test, y_test, learning_rate=0.1):
    """
    ExtraTrees
    """
    import xgboost as xgb
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    LOGGER.info('start predict: n_trees={},X_train.shape={},y_train.shape={},X_test.shape={},y_test.shape={}'.format(
        n_trees, X_train.shape, y_train.shape, X_test.shape, y_test.shape))
    clf = xgb.XGBClassifier(n_estimators=n_trees, max_depth=max_depth, objective='multi:softprob',
            seed=0, silent=True, nthread=-1, learning_rate=learning_rate)
    eval_set = [(X_test, y_test)]
    clf.fit(X_train, y_train, eval_set=eval_set, eval_metric="merror")
    y_pred = clf.predict(X_test)
    prec = float(np.sum(y_pred == y_test)) / len(y_test)
    LOGGER.info('prec_xgb_{}={:.6f}%'.format(n_trees, prec*100.0))
    return clf, y_pred


# 逻辑回归
def prec_log(X_train, y_train, X_test, y_test):
    from sklearn.linear_model import LogisticRegression
    if not issparse(X_train):
        X_train = X_train.reshape((X_train.shape[0], -1))
    if not issparse(X_test):
        X_test = X_test.reshape((X_test.shape[0], -1))
    LOGGER.info('start predict: X_train.shape={},y_train.shape={},X_test.shape={},y_test.shape={}'.format(
        X_train.shape, y_train.shape, X_test.shape, y_test.shape))
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    clf = LogisticRegression(solver='sag', n_jobs=-1, verbose=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    prec = float(np.sum(y_pred == y_test)) / len(y_test)
    LOGGER.info('prec_log={:.6f}%'.format(prec*100.0))
    return clf, y_pred


#  绘图
def plot_forest_all_proba(y_proba_all, y_gt):
    from matplotlib import pylab
    N = len(y_gt)
    num_tree = len(y_proba_all)
    pylab.clf()
    mat = np.zeros((num_tree, N))
    LOGGER.info('mat.shape={}'.format(mat.shape))
    for i in range(num_tree):
        mat[i,:] = y_proba_all[i][(range(N), y_gt)]
    pylab.matshow(mat, fignum=False, cmap='Blues', vmin=0, vmax=1.0)
    pylab.grid(False)
    pylab.show()


# 绘矩阵
def plot_confusion_matrix(cm, label_list, title='Confusion matrix', cmap=None):
    from matplotlib import pylab
    cm = np.asarray(cm, dtype=np.float32)
    for i, row in enumerate(cm):
        cm[i] = cm[i] / np.sum(cm[i])
    #import matplotlib.pyplot as plt
    #plt.ion()
    pylab.clf()
    pylab.matshow(cm, fignum=False, cmap='Blues', vmin=0, vmax=1.0)
    ax = pylab.axes()
    ax.set_xticks(range(len(label_list)))
    ax.set_xticklabels(label_list, rotation='vertical')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_yticks(range(len(label_list)))
    ax.set_yticklabels(label_list)
    pylab.title(title)
    pylab.colorbar()
    pylab.grid(False)
    pylab.xlabel('Predicted class')
    pylab.ylabel('True class')
    pylab.grid(False)
    pylab.savefig('test.jpg')
    pylab.show()
