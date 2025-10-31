import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay
import os
# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.


def plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies):
	# TODO: Make plots for loss curves and accuracy curves.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	n = len(train_losses)
	if not ((valid_losses) == (train_accuracies) == (valid_accuracies) == n):
		n = min(n, len(valid_losses), len(train_accuracies), len(valid_accuracies))
		train_losses = train_losses[:n]
		valid_losses = valid_losses[:n]
		train_accuracies = train_accuracies[:n]
		valid_accuracies = valid_accuracies[:n]
		
	epochs = np.arange(1, n + 1)
	fig, (ax1,ax2) = plt.subplots(1,2, figsize =(10,8))
	ax1.set_title('Loss Curves')
	ax1.plot(epochs, train_losses, label ='Training Loss')
	ax1.plot(epochs, valid_losses, label = 'Validation Loss')
	ax1.legend(loc = "upper right")
	ax1.set_xlabel("epoch")
	ax1.set_ylabel("Loss")
	ax2.set_title('Accuracy curves')
	ax2.plot(epochs, train_accuracies,  label = 'Training Accuracy')
	ax2.plot(epochs, valid_accuracies,  label ='Validation Accuracy')
	ax2.legend(loc = "upper left")
	ax2.set_xlabel("epoch")
	ax2.set_ylabel("Accuracy")
	plt.tight_layout()
	os.makedirs("imgs", exist_ok=True)
	out_path = os.path.join("imgs", "learning_curves_RNN_varimpr.png")
	fig.savefig(out_path, dpi=200, bbox_inches="tight")
	plt.close()

def plot_confusion_matrix(results, class_names, normalize=True):
	# TODO: Make a confusion matrix plot.
	# TODO: You do not have to return the plots.
	# TODO: You can save plots as files by codes here or an interactive way according to your preference.
	#refrence https://www.geeksforgeeks.org/machine-learning/how-to-plot-confusion-matrix-with-labels-in-sklearn/
	y_test = [t for t, _ in results]
	y_pred = [p for _, p in results]
    
	
	cm = confusion_matrix(y_test, y_pred)
	cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
	disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= class_names)
	fig, ax = plt.subplots(figsize=(7,6))
	disp.plot(include_values=True,  cmap="Blues", ax=ax, colorbar= True)
	ax.set_xlabel("Predicted")
	ax.set_ylabel("True")
	ax.set_title("Normalized conusion Matrix")
	plt.tight_layout()
	os.makedirs("imgs", exist_ok=True)
	out_path = os.path.join("imgs", "confusion_matrix_RNN_varimpr.png")
	fig.savefig(out_path, dpi=200, bbox_inches="tight")
	plt.close()