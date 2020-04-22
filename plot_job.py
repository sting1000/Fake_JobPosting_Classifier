import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, plot_precision_recall_curve
from sklearn.metrics import roc_curve, auc, roc_auc_score
sns.set(style="whitegrid")


def plot_cm(model, X_test, y_test):
    """
    function to visualize confusion matrix
    :param model: a trained model, support Classifier and NN
    :param X_test: dataframe
    :param y_test: dataframe with one lable column
    :return: None
    """
    try:
        # if model is a classifier
        disp = plot_confusion_matrix(model, X_test, y_test,
                                     cmap=plt.cm.Blues,
                                     normalize=None)
        cm = disp.confusion_matrix
    except:
        # get cm if model is a NN model
        predictions = model.predict(X_test)
        predictions = np.round(predictions).astype(int)
        cm = tf.Session().run(
            tf.math.confusion_matrix(labels=np.ravel(y_test), predictions=predictions))
        sns.heatmap(cm, annot=True)

    tn, fp, fn, tp = cm.ravel()

    # calculate score
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    F1 = 2 * recall * precision / (recall + precision)

    # print and show plot
    print('Recall={0:0.3f}'.format(recall), '\nPrecision={0:0.3f}'.format(precision))
    print('F1={0:0.3f}'.format(F1))
    plt.show()


def plot_aucprc(model, X_test, y_test):
    """
    Function to draw AUC plot
    :param model: a trained model, support Classifier and NN
    :param X_test: dataframe
    :param y_test: dataframe with one lable column
    :return:
    """
   

    # draw prc
    
    try:
        scores = model.decision_function(X_test)
        average_precision = average_precision_score(y_test, scores)
        disp = plot_precision_recall_curve(model, X_test, y_test)
        disp.ax_.set_title('Precision-Recall curve: '
                        'AP={0:0.2f}'.format(average_precision))
    except:
        scores = model.predict(X_test)
        average_precision = average_precision_score(y_test, scores)

    print('Precision-Recall curve: '
                        'AP={0:0.2f}'.format(average_precision))
    plt.show()

    # draw roc
    fpr, tpr, _ = roc_curve(np.ravel(y_test), scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw,
            label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: AUC={0:0.2f}'.format(roc_auc))
    plt.legend(loc="lower right")
    plt.show()


def add_accum(df, col):
    """
    add a accumulated column to dataframe with prefix "accum_"
    :param df: dataframe
    :param col: string, name of the column to be accumulated
    :return: dataframe with new column
    """
    accumlator = 0
    accum_list = []
    for value in df[col]:
        accumlator += value
        accum_list.append(accumlator)
    df["accum_" + col] = accum_list
    df["accum_" + col] /= accum_list[-1]
    return df


def draw_accum(ax, df, cols):
    """
    draw the accumulated plot using specified cols
    :param ax: plt.ax, the axis to plot on
    :param df: dataframe
    :param cols: list of string, which columns are in plot
    :return: None
    """
    df = df.reset_index()
    df["index"] = df.index
    for col in cols:
        df = add_accum(df, col)
        acc_col_name = "accum_" + col
        ax.plot("index", acc_col_name,
                data=df[["index", acc_col_name]],
                label=col
                )
    ax.legend()
    ax.set(ylabel="Percentage", xlabel="Top n catagories", title="Accummulated Lineplot")


def draw_bar(ax, df, x, y, xlim, head=20):
    # Plot the total crashes
    sns.set_color_codes("pastel")
    sns.barplot(x="total_count",
                y=y,
                data=df[:head],
                label="Total Count",
                color="b",
                ax=ax)

    # Plot the crashes where alcohol was involved
    sns.set_color_codes("muted")
    sns.barplot(x=x,
                y=y,
                data=df[:head],
                label=x,
                color="b",
                ax=ax)

    # Add a legend and informative axis label
    ax.legend(ncol=2, loc="lower right", frameon=True)
    ax.set(xlim=xlim, ylabel=y, xlabel="count", title=y + "-" + x + "ratio comparison")
    sns.despine(left=True, bottom=True, ax=ax)


def plotCateColumn(data, ratio_col, group_by, xlim_bar, sort_by='ratio', ascending=False):
    """
    This function is to visualize data groupby y, regarding binary label x.
    :param data: the dataframe including columns ratio_col, group_by
    :param ratio_col: The string name of label column
    :param group_by: The category for groupby (analysis)
    :param xlim: the range of x axis
    :param sort_by: the method for sort. two options: 'ratio', 'total_count'
    :param head: the number of rows to display
    :return: 
    """
    # create dataframe not modify the source file
    temp = data.copy()
    temp["total_count"] = 1

    # calculate taget column ratio
    temp = temp.groupby(group_by).sum()
    temp["ratio"] = temp[ratio_col] / temp["total_count"]
    temp = temp.reset_index().sort_values(by=sort_by, ascending=False)
    temp = temp[[group_by, ratio_col, 'total_count', 'ratio']]

    # Initialize the matplotlib figure
    fig = plt.figure(figsize=(16, 10))
    grid = plt.GridSpec(2, 1, hspace=0.4)
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[1, 0])

    # use function to draw on ax
    draw_accum(ax=ax1, df=temp, cols=["fraudulent", "total_count"])
    draw_bar(ax=ax2, df=temp, y=group_by, x=ratio_col, xlim=xlim_bar)

    plt.show()
    return temp
