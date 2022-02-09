# General functions:

def denormalize_betas(lr, scaler):
    """
    gets logistic regression classifer and scaler, and returns the coefficent of the de-normalized features. 
    lr - sklearn logistic regression classifer
    scaler - StandardScaler with mean and scale.
    """
    beta_vec = lr.coef_[0]/scaler.scale_
    intercept = lr.intercept_[0] - np.dot(scaler.mean_/scaler.scale_,lr.coef_[0])
    return beta_vec, intercept


def plot_predicated_probability(clf, X=X_train, y=y_train, title = "Predicted probability of survival", y_label=y_label):
    """ plots the distribuation of predicated probability by the clf on X, given the actual class is 1 or 0 """
    
    # define subplots:
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(17,10))
    index, ax = 0, ax.flatten()
    
    # set title
    fig.suptitle(f"{title} - {clf.__class__.__name__}", fontsize=20, weight="bold")
    
    sns.histplot(clf.predict_proba(X[y[y_label] == 1])[:,1], ax=ax[0], color="gold", stat="probability").set_title("actual=survived", fontsize=20, weight="normal")
    sns.histplot(clf.predict_proba(X[y[y_label] == 0])[:,1], color="salmon", stat="probability").set_title("actual=not survived", fontsize=20, weight="normal")


def plot_roc_curve(clf, color=color_main, train=True, ax=None, benchmarks={}, label='', X=None, y=None):
    """ plots the ROC curve of the model """
    if (X is None) or (y is None):
        if train:
            X,y = X_train, y_train
        else:
            X,y = X_test, y_test

    y_proba = clf.predict_proba(X)[:,1]
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    if ax is not None:
        sns.lineplot(x=fpr, y=tpr, color=color, ci=None, ax=ax,label=label)
        fig = sns.lineplot(x=thresholds, y=thresholds, color=color_main, ci=None, linestyle = '--', ax=ax)
    else:
        sns.lineplot(x=fpr, y=tpr, color=color, ci=None,label=label)
        fig = sns.lineplot(x=thresholds, y=thresholds, color=color_main, ci=None, linestyle = '--')
        
    fig.set_title("ROC curve")
    fig.set(xlim=(0, 1), ylim = (0,1))
    plt.xlabel('False positive rate (FP / N)')
    plt.ylabel('True positive rate (recall)')    


def evaluate(clf, train=True, feature_importance=False):
    """ prints and plots a custom score report for a sklearn classifer"""
    class_names = ["Died", "Survived"]
    if train:
        X,y = X_train, y_train
    else:
        X,y = X_test, y_test
        
    # predict     
    y_predict = clf.predict(X)
    
    name = clf.__class__.__name__
    
    cols = X.columns
    
    # score report
    print(classification_report(y, y_predict, target_names=class_names))
    
    # balance in prediction report
    print(f"""mean_of_predict={y_predict.mean():.2f} (actual = {y.values.mean():.2f}%)""")
    
    if feature_importance:
        # feature importance plot
        typey = "train" if train else "test"
        title = f"features importances for {name}, {typey}"
        if name == 'LogisticRegression':
            feat_importances = pd.Series(clf.coef_[0], index=cols)
        else:
            feat_importances = pd.Series(clf.feature_importances_, index=cols)
        feat_importances.sort_values(ascending=True).plot(kind='barh',color=color_main)
        plt.suptitle(title, fontsize=20, weight="bold")

    return f1_score(y, y_predict)

def plot_scores(clf, train=True, ax=None, n_thresholds = 20, X=None, y=None):
    """ plots f1, recall and precision for different thresholds of the probabilities 
        returns the best threshold """
    if (X is None) or (y is None):
        if train:
            X,y = X_train, y_train
        else:
            X,y = X_test, y_test
       
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

def plot_tree(classifier, max_depth=4):
    """ plots one tree, up to max_depth """
    fn=X_train.columns
    
    bool2str = lambda x:"Survived" if x==1.0 else "Dies"
    cn = np.array([bool2str(xi) for xi in classifier.classes_])

    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=max(200*max_depth,300))  #make image clearer than default

    tree.plot_tree(classifier,
               feature_names = fn, max_depth=max_depth,
               class_names=cn,
               filled = True)

