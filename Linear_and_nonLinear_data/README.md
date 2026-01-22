# Advanced Machine Learning – Kernel Methods

This repository demonstrates core concepts behind **Advanced Machine Learning**, with a strong focus on **linearity, non-linearity, and kernel methods**, particularly in the context of **Support Vector Machines (SVMs)**.

The goal of this project is to build an intuitive and practical understanding of *why* linear models fail on certain datasets and *how* kernel methods allow us to overcome these limitations without abandoning linear learning algorithms.

---

## Motivation

Many classical machine learning models assume that the relationship between input features and the output is **linear**. While this assumption simplifies learning and interpretation, real-world data often violates it. Understanding when linear models work, when they fail, and how to extend them to handle non-linear patterns is a fundamental step toward advanced machine learning.

This repository explores these ideas using synthetic datasets and visualizations, culminating in the use of **kernelized SVMs** to solve non-linearly separable problems.

---

## Linearity

**Linearity** means that the relationship between input features and the output can be represented using a straight line (or a flat surface in higher dimensions).

* Increasing or decreasing an input feature affects the output in a **proportional and predictable** way.
* For classification, this implies that classes can be separated by:

  * a straight line in 2D
  * a plane in 3D
  * a hyperplane in higher dimensions

Examples of linear models include:

* Linear Regression
* Logistic Regression
* Linear Support Vector Machines (SVMs)

These models work well **only when the underlying structure of the data is linearly separable**.

---

## Non-Linearity

**Non-linearity** arises when the relationship between inputs and outputs cannot be captured by a straight line.

* The influence of one feature may depend on the value of another
* Decision boundaries may be curved or more complex
* No single straight line can correctly separate the classes

A classic example is the **XOR problem**, where data points from different classes are arranged diagonally. In this case:

* Linear models completely fail
* No linear boundary can separate the classes correctly

This highlights a critical limitation of purely linear approaches.

---

## Why Linear Models Fail on Non-Linear Data

Linear models lack the **capacity** to bend or adapt their decision boundaries.

When data is non-linear:

* Linear classifiers underfit
* Increasing training time or tuning parameters does not help
* The model simply cannot represent the required decision surface

To solve this, we need a way to **enrich the feature representation** of the data.

---

## Kernel Methods: The Key Idea

Instead of abandoning linear models entirely, kernel methods allow us to **extend them intelligently**.

The core idea is:

> Transform the original data into a higher-dimensional feature space where a linear separator *does* exist.

Key points:

* The transformation is **implicit**, not computed explicitly
* Linear models are still used in the transformed space
* The resulting decision boundary becomes **non-linear** in the original space

Support Vector Machines (SVMs) achieve this using **kernel functions**.

---

## Kernels and Geometry

A **kernel** defines how similarity between data points is measured.

By choosing different kernels, we:

* Change the geometry of the feature space
* Alter how distances and similarities are computed
* Enable linear separation of previously non-linear patterns

Common kernels explored in this repository include:

* **Linear Kernel** – fails on non-linear data
* **Polynomial Kernel** – captures feature interactions
* **RBF (Gaussian) Kernel** – highly flexible and powerful

Although the SVM learns a linear boundary in the transformed space, this boundary corresponds to a **curved, non-linear boundary** in the original input space.

---

## Learning Process Summary

The workflow demonstrated in this repository follows a clear conceptual progression:

1. Understand whether the data is **linear or non-linear**
2. Observe how **linear models fail** on non-linear patterns
3. Apply **kernel methods** to transform the data
4. Achieve **linear separability in a higher-dimensional space**

This is the core idea behind kernelized SVMs.

---

##  Repository Structure

```
lab1/
├── data_generation.py         # Synthetic data generators (linear, XOR)
├── visualization.py           # 2D plotting utilities
├── kernel_methods.py          # Linear vs kernel SVM vs polynomial SVM experiments
├── visualising_linear_data.py # to visualise Linearly seperable data
├── visualization.py           # to visualise Linear data
├── XOR_data_visualization.py  # to visualise/ see non linearly seperable data
├── README.md
```

---

## Learning Outcomes

By working through this repository, you will:

* Develop intuition about linear vs non-linear data
* Understand why linear models have fundamental limits
* Learn how kernel methods overcome these limits
* Gain hands-on experience with SVM kernels and visualization

---

## Conclusion

Kernel methods provide a powerful bridge between simplicity and flexibility. They allow us to retain the mathematical elegance of linear models while extending their applicability to complex, real-world patterns.

This repository serves as a practical and conceptual guide to understanding that bridge.

---

