import matplotlib.pyplot as plt
import seaborn           as sns
import numpy as np
import pandas as pd

def plot_confusion_matrix(cm,target_names):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    fig, ax = plt.subplots(figsize=(7,6))
    tick_marks = np.arange(len(target_names)) + 0.5

    # create heatmap
    sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu", fmt='g')
#    ax.xaxis.set_label_position("top")
    plt.title('Confusion matrix') #, y=1.1
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.xlabel('Predicted \naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(["Not Default", "Default"])
    ax.set_yticklabels(["Not Default", "Default"])
#    plt.xticks(tick_marks, ["Not Default", "Default"])
#    plt.yticks(tick_marks, ["Not Default", "Default"])
    return fig