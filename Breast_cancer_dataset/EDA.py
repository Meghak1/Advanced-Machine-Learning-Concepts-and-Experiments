import matplotlib.pyplot as plt
import seaborn as sns


def perform_eda(X, y):
    print("Exploratory data analysis for given breast cancer dataset")

    # Dataset shape
    print("Number of samples:", X.shape[0])
    print("Number of features:", X.shape[1])
    print("\nClass distribution:")
    print(y.value_counts())

    # Missing values
    print("\nMissing values:")
    print(X.isnull().sum().sum())

    # Statistical summary
    print("\nStatistical summary:")
    print(X.describe())

    # Target distribution plot
    plt.figure()
    y.value_counts().plot(kind="bar")
    plt.title("Target Class Distribution")
    plt.xlabel("Class (0 = Malignant, 1 = Benign)")
    plt.ylabel("Count")
    plt.show()

    # Correlation heatmap (first 10 features for clarity)
    plt.figure(figsize=(10, 8))
    sns.heatmap(X.iloc[:, :10].corr(), cmap="coolwarm")
    plt.title("Feature Correlation Heatmap (First 10 Features)")
    plt.show()
