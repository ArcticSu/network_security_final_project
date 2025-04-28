import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from dataloader import load_dataset

def flatten_and_normalize(X):
    direction = X[:, :, 0:1]
    deltaT = X[:, :, 1:2]
    size = X[:, :, 2:3]

    deltaT = (deltaT - deltaT.min()) / (deltaT.max() - deltaT.min() + 1e-8)
    size = (size - size.min()) / (size.max() - size.min() + 1e-8)

    X_normalized = np.concatenate([direction, deltaT, size], axis=-1)
    return X_normalized.reshape((X.shape[0], -1))

def main():
    X, Y, encoder = load_dataset("processed_file", window_size=300, stride=50)
    X = flatten_and_normalize(X) 

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    print("Training SVM model...")
    clf = SVC(kernel='rbf', C=1.0, gamma='scale')  
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    # cm = confusion_matrix(y_test, y_pred)
    # plt.figure(figsize=(6, 5))
    # sns.heatmap(cm, annot=True, fmt="d", xticklabels=encoder.classes_, yticklabels=encoder.classes_, cmap="Blues")
    # plt.title("SVM Confusion Matrix")
    # plt.xlabel("Predicted")
    # plt.ylabel("True")
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()
