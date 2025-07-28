import glob
import cv2
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

IMG_SIZE = (64, 64)


# Define label mapping
label_map = {'cat': 0, 'dog': 1}

def data_preprocess(root_path):
    data = []
    labels = []
    print(f'Processing data from {root_path}...')
    for i, address in enumerate(glob.glob(f'{root_path}\\*\\*')):
        img = cv2.imread(address)
        if img is None:
            continue

        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0
        img = img.flatten()
        data.append(img)

        label = address.split('\\')[-2].lower()
        labels.append(label_map[label])

        if i % 100 == 0:
            print(f'[INFO] {i} images processed...')

    return np.array(data), np.array(labels)


print("Loading training data...")
X_train, y_train = data_preprocess('train')

print("Loading test data...")
X_test, y_test = data_preprocess('test')

# SGDClassifier
print("\nSGDClassifier Training...")
sgd_model = SGDClassifier(random_state=42)
sgd_model.fit(X_train, y_train)
y_pred_sgd = sgd_model.predict(X_test)
print(f"SGDClassifier Accuracy: {accuracy_score(y_test, y_pred_sgd):.4f}")

# KNN Classifier
print("\nKNN Training...")
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
print(f"KNN Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")


print("In this experiment, in addition to the original 5 test images of cats and 5 test images of dogs provided in the test folder,")
print("I also downloaded 10 more images of cats and 10 more images of dogs from the internet and added them to the test folder.")
print("Also, I experimented with adjusting the parameters of KNN (changing the n_neighbors value) and SGDClassifier (changing the random_state value) to observe the effect on results.")
print("The test accuracy I obtained generally ranged from 0.45 to 0.65.")
print("There are a few reasons for this moderate accuracy. First, the number of test images is relatively small, which can make the accuracy more sensitive to individual misclassifications.")
print("Second, images collected from the internet may have different styles, backgrounds, or resolutions compared to the training data, which can make them harder for the model to classify correctly.")