from sklearn.svm import SVC
from data_generation import generate_linear_data
from visualization import plot_data, plot_svm

X, y = generate_linear_data()
plot_data(X, y, "Linearly Separable Data")

svm_hard = SVC(kernel='linear', C=1e6)
svm_hard.fit(X, y)

print("Number of support vectors : ", len(svm_hard.support_vectors_))
plot_svm(X, y, svm_hard, "Hard Margin SVM")