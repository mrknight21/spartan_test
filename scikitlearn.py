# Compiled from https://scikit-learn.org/stable/tutorial/basic/tutorial.html
# Modified to save image to disk

from sklearn import datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()
digits = datasets.load_digits()

print(digits.data)

# Shape of the data arrays
digits.images[0]

# Consider the estimator as a black box
from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)

# Choosing the parameters of the model
clf.fit(digits.data[:-1], digits.target[:-1])

# predict new values. 
# clf is your scikit-learn classifier and digits is your dataset

predicted_label = clf.predict(digits.data[-1:])
save_image = digits.data[-1].reshape(8, 8)  # Reshape the data to an image-like format

# Display the image using matplotlib
plt.imshow(save_image, cmap=plt.cm.gray)
plt.title(f"Predicted Label: {predicted_label[0]}")
plt.savefig('predicted_image.png')  # Save the image to a file

# If done with X-Windows forwarding or OOD
# plt.show()


