from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, labels, filename):

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
    disp.plot()
    # Attempt to directly modify text elements if available
    if hasattr(disp, 'text_') and disp.text_ is not None:
        for texts in disp.text_:
            for text in texts:
                text.set_fontsize(20)  # Adjust fontsize as needed
    else:
        # Fallback: Iterate over ax children and find Text objects
        for child in ax.get_children():
            if isinstance(child, matplotlib.text.Text):
                child.set_fontsize(20)  # Adjust fontsize as needed


    # Save the figure
    plt.savefig(filename, dpi=300, format='jpeg')
    plt.show()