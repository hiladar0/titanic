# General functions:

# data analysis and wrangling
import pandas as pd
import numpy as np
from scipy import stats

# visualization
from IPython.display import Image
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_score,roc_curve,roc_auc_score, recall_score, accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# plot settings:
CMAP = 'vlag'
fontfamily = "monospace"
context = "notebook"

color_main = 'gold'
my_colors = [color_main, 'salmon', 'wheat', 'red']


rc = {'axes.titlesize':18, 'axes.titleweight':'bold',
      'axes.labelweight':'bold', 'figure.figsize':[12,8], 
      'lines.linewidth':5, 'axes.edgecolor':'black',
     'boxplot.boxprops.color': color_main}

sns.set(context=context, font=fontfamily, rc=rc, style="whitegrid")
sns.set_palette(CMAP)

def denormalize_betas(lr, scaler):
    """
    gets logistic regression classifer and scaler, and returns the coefficent of the de-normalized features. 
    lr - sklearn logistic regression classifer
    scaler - StandardScaler with mean and scale.
    """
    beta_vec = lr.coef_[0]/scaler.scale_
    intercept = lr.intercept_[0] - np.dot(scaler.mean_/scaler.scale_,lr.coef_[0])
    return beta_vec, intercept


def plot_predicated_probability(clf, X, y,y_label, title = "Predicted probability of survival"):
    """ plots the distribuation of predicated probability by the clf on X, given the actual class is 1 or 0 """
    
    # define subplots:
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(17,10))
    index, ax = 0, ax.flatten()
    
    # set title
    fig.suptitle(f"{title} - {clf.__class__.__name__}", fontsize=20, weight="bold")
    
    sns.histplot(clf.predict_proba(X[y[y_label] == 1])[:,1], ax=ax[0], color="gold", stat="probability").set_title("actual=survived", fontsize=20, weight="normal")
    sns.histplot(clf.predict_proba(X[y[y_label] == 0])[:,1], color="salmon", stat="probability").set_title("actual=not survived", fontsize=20, weight="normal")


def plot_roc_curve(clf,X,y, color=color_main, ax=None, label=''):
    """ plots the ROC curve of the model """
    y_proba = clf.predict_proba(X)[:,1]
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    if ax is not None:
        sns.lineplot(x=fpr, y=tpr, color=color, ci=None, ax=ax,label=label)
        fig = sns.lineplot(x=thresholds, y=thresholds, color=color_main, ci=None, linestyle = '--', ax=ax)
    else:
        sns.lineplot(x=fpr, y=tpr, color=color, ci=None,label=label)
        fig = sns.lineplot(x=thresholds, y=thresholds, color='black', ci=None, linestyle = '--')
        
    fig.set_title("ROC curve")
    fig.set(xlim=(0, 1), ylim = (0,1))
    plt.xlabel('False positive rate (FP / N)')
    plt.ylabel('True positive rate (recall)')    


def evaluate(clf, X, y, feature_importance=False, features_names=[], class_names=[]):
    """
    evaluate sklearn classifer.

        Parameters:
                clf (sklearn.classifier): the classifier
                X (pandas.dataframe or numpy.array) feature matrix
                y (pandas.dataframe or numpy.array)
                feature_importance (bool) flag if to plot feature importance plot
                features_names - list of names
                class_names - list of y_label classes

        Returns:
                f1 score
    """
    # predict     
    y_predict = clf.predict(X)
    
    name = clf.__class__.__name__
        
    # score report
    if len(class_names) > 0:
        print(classification_report(y, y_predict, target_names=class_names))
    else:
        print(classification_report(y, y_predict))
    
    # balance in prediction report
    print(f"""mean_of_predict={y_predict.mean():.2f} (actual = {y.values.mean():.2f}%)""")
    
    if feature_importance:
        # feature importance plot
        title = f"features importances for {name}"
        if name == 'LogisticRegression':
            if len(features_names) > 0:
                feat_importances = pd.Series(clf.coef_[0], index=features_names)
            else:
                feat_importances = pd.Series(clf.coef_[0])
        elif name == 'RandomForestClassifier':
            if len(features_names) > 0:
                feat_importances = pd.Series(clf.feature_importances_, index=features_names)
            else:
                feat_importances = pd.Series(clf.feature_importances_)
            
        feat_importances.sort_values(ascending=True).plot(kind='barh',color=color_main)
        plt.suptitle(title, fontsize=20, weight="bold")

    return f1_score(y, y_predict)
    
def regression_statistical_results_summary(sm_model,features):
    """take the result of an statsmodel results table and transforms it into a dataframe"""
    pvals = np.round(sm_model.pvalues,3)
    coeff = sm_model.params
    cols = ['const']+features
    results_df = pd.DataFrame(data={"coeff":coeff,"pvals":pvals, "cols":cols}).set_index('cols').sort_values('pvals', ascending=False)
    return results_df    

def plot_scores(clf, X, y, ax=None, n_thresholds = 20):
    """ plots f1, recall and precision for different thresholds of the probabilities 
        returns the best threshold """
       
    y_proba = clf.predict_proba(X)[:,1]
    rows = []
    thresholds = np.arange(0,0.99,1/n_thresholds)
    for thr in thresholds:
        if thr < 0.99:
            pred = (y_proba>thr).astype(int)
            rows.append([thr, f1_score(y,pred ), precision_score(y,pred ), recall_score(y,pred )])

    scores_by_thr = pd.DataFrame(rows, columns = ["thresholds","f1_score","precision_score","recall_score"])
    if ax is not None:
        fig = sns.lineplot(data=scores_by_thr, x='thresholds', y='f1_score', ci=None,color=color_main, label ="f1_score", ax=ax)
    else:
        fig = sns.lineplot(data=scores_by_thr, x='thresholds', y='f1_score', ci=None,color=color_main, label ="f1_score")
    sns.lineplot(data=scores_by_thr, x='thresholds', y='recall_score', ci=None,color=my_colors[3], label = "recall_score")
    sns.lineplot(data=scores_by_thr, x='thresholds', y='precision_score', ci=None,color=my_colors[1], label = "precision_score")

    fig.set(xlim=(0, 1), ylim = (0,1))
    fig.set_title("Model score as a function of prediction threshold", fontsize=20)
    plt.xlabel('threshold')
    plt.ylabel('Score')
    return scores_by_thr.sort_values("f1_score", ascending=False).iloc[0]['thresholds']

def plot_tree(classifier, feature_names, max_depth=4):
    """ plots one tree, up to max_depth """    
    bool2str = lambda x:"Survived" if x==1.0 else "Dies"
    cn = np.array([bool2str(xi) for xi in classifier.classes_])

    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=max(200*max_depth,300))  #make image clearer than default

    tree.plot_tree(classifier,
               feature_names = feature_names, max_depth=max_depth,
               class_names=cn,
               filled = True)

