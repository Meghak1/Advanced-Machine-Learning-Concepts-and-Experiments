from sklearn.svm import SVC
from data_generation import generate_overlap_data
from visualization import plot_svm

X, y = generate_overlap_data()
svm_soft = SVC(kernel='linear', C=0.1)
svm_soft.fit(X, y)

print("Number of support vectors : ", len(svm_soft.support_vectors_))
plot_svm(X, y, svm_soft, "Soft Margin SVM C=0.1")