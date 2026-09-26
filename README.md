# Examining the Effectiveness of PCA-Based Pooling in Convolutional Neural Networks

### An Experimental Study Using Simple CNN, LeNet, and ResNet-18 on MNIST

## Abstract
Pooling layers in convolutional neural networks (CNNs) is a common downsampling method to reduce spatial dimension and computional requrements. However, traditional pooling methods can discard informatinfrom the pooling region. Principal component analysis (PCA) based pooling can be an alternative that attempts to preserve important information through principal component analysis (PCA) [1].

This study conduct a comprehensive analysis of PCA-based pooling compared with conventional max pooling.
Three CNN architectures—Simple CNN, LeNet, and ResNet-18—are evaluated on the MNIST handwritten digit classification dataset. The comparison considers classification accuracy, Macro F1-score, trainable parameters, FLOPs, training time, inference time, GPU memory consumption, and convergence behavior.

The experimet shows that PCA-based pooling improves the accuracy of the Simple CNN and ResNet-18 by 0.34 and 0.07 percentage points, respectively, while reducing the accuracy of LeNet by 0.31 percentage points. However, PCA-based pooling introduces additional computational and memory overhead. These results suggest that PCA-based pooling should be evaluated not only in terms of accuracy but also in terms of its computational cost and architecture-specific behavior.

---

## 1. Introduction

Convolutional neural networks (CNNs) has now become a fundamental approach for image classification and visual recognition [2]. A typical CNN consists of convolutional layers, nonlinear activation functions, pooling or downsampling operations, and classification layers. Pooling in CNNs reduces the spatial dimension of feature maps and can help reduce computational requirements while preserving the important informatioon.

MaxPool is most commonly used pooling operation. It picks the maximum activation within a local region:

$$
y_{\text{max}}=\max(x_1,x_2,\ldots,x_d)
$$

Although this operation preserves the strongest activation, it may discard the remaining values in the pooling region. This can result in information loss.

To address this limitation, Zhao et al. proposed **PCAPool**, a pooling method based on Principal Component Analysis (PCA), the objective of this is to retain more information from the local pooling region [1]. Their mathod construct a sample matrix from the feature maps, the applies PCA operation, use the principal components to generate the pooling representation.

PCA is a classical dimentionality reduction technique that transform core correlated variables into orthogonal components ordered in the direction of varience they preserve[3], [4]. The first principal component captures the direction of maximum variance.



The main research question is:

> **Does PCA-based pooling prevent information loss that increase accuracy compared with max pooling when computational and memory costs are also considered?**

---

## 2. Related Work

### 2.1 Principal Component Analysis

Principal Component Analysis is a statistical method for transforming correlated variables into principal components [4].

For a centered data matrix \(X\), the covariance matrix is

$$
\Sigma=\frac{1}{n-1}X^TX.
$$

The principal components are obtained from the eigenvalue decomposition

$$
\Sigma v_i=\lambda_i v_i,
$$

where, $v_i$ is an eigenvector and $\lambda_i$ is its corresponding eigenvalue. The largest eigenvectors represents the direction of maximum variance.

### 2.2 PCA-Based Pooling

Zhao et al. proposed **PCAPool**, they applies PCA operation to replace the traditional pooling representation [1]. Their experiments investigated PCAPool using several CNN architectures and datasets, including MNIST, CIFAR-10, CIFAR-100, SVHN, and ImageNet. This work is the motivation to use PCA as downsampling method.
PCA is a popular dimentionality reduction method. For deep learning, there are  other studies have also investigated the use of PCA. For example, PCANet uses cascaded PCA to learn convolutional filter banks for image classification [5]. PCA has also been used for CNN compression and feature representation [6].

However, PCA-based operation is associated with additional  computational overhead. Therefore, an accuracy comparison alone may not fully characterize the effectiveness PCA-based pooling.

---

## 3. PCA-Based Pooling Formulation
The steps of PCA operation are as following:

#### 1. Sample Matrix

A sliding window travarse throuth all the feature and form a sample matrix:

$$
A =
\begin{bmatrix}
\boldsymbol{\alpha}_1^T\\
\boldsymbol{\alpha}_2^T\\
\vdots\\
\boldsymbol{\alpha}_k^T
\end{bmatrix}
\in \mathbb{R}^{k \times d}
$$

where:

- $k$ = number of samples (rows).
- $d$ = number of features in each flattened sample.
- $\boldsymbol{\alpha}_i \in \mathbb{R}^d$ = the $i$-th sample vector.


#### 2. Calculate the Sample Mean:
$$
\bar{\mathbf{\alpha}} = \frac{1}{k} \sum_{i=1}^{k} \mathbf{\alpha}_i
$$

$$
\bar{\boldsymbol{\alpha}}
=
\frac{1}{k}
\sum_{i=1}^{k}
\boldsymbol{\alpha}_i
$$


The difference between the row vector and the sample mean is used to obtain the matrix:

$$
A_c =
\begin{bmatrix}
(\boldsymbol{\alpha}_1-\bar{\boldsymbol{\alpha}})^T\\
(\boldsymbol{\alpha}_2-\bar{\boldsymbol{\alpha}})^T\\
\vdots\\
(\boldsymbol{\alpha}_k-\bar{\boldsymbol{\alpha}})^T
\end{bmatrix}
$$

#### 3. The eigenmatrix is obtained by singular value decomposition of the sample matrix a:
$$
A_c = U\Sigma V^T
$$

Extract the principal component matrix from the sample matrix A.

$$
Z = A_c V
$$

After PCA projection, PCAPool applies information weighting to the principal components:

$$
\mathbf{p} = \sum_{j=1}^{r}\gamma_j Z_{:,j}
$$

#### 4. Block Arrangement and Pooling Output

The last step is to convert pooling vector into desired number of  the pooling result that is the feature maps after dimensionality reduction.

## 4. Experimental Setup

### 4.1 Dataset

The experiments are conducted on the **MNIST handwritten digit dataset**, consisting ten digit classes of grayscale images.

### 4.2 CNN Architectures

Three architectures are considered:

1. **Baseline CNN (BCNN)** — a lightweight simple custom CNN used as a controlled baseline.
2. **LeNet** — a classical CNN architecture originally developed for handwritten digit recognition [7].
3. **ResNet-18** — a residual CNN architecture based on residual learning [8].

For each architecture, PCAPool is comppared against maxpool. For the sake of the experiment all the other parameters are configured exectly same for both architecture.

### 4.3 Evaluation Metrics
To examine the characteristics the following metrics are used:

* Top-1 classification accuracy
* Macro F1-score
* Trainable parameters
* FLOPs
* Training time
* Inference time
* Inference time per image
* Peak GPU memory
* Convergence epoch

Maxpool preserve strong activation, PCAPool preserve global stucture. The goal is to examine under which conditiona nd scenario PCAPool works so that we can determine the use.
---

## 5. Experimental Results

| Model      | Pooling | Accuracy (%) |   Macro F1 | Params (M) | FLOPs (M) | Train Time (s) | Inference (ms/img) | Peak GPU Mem. (MB) | Conv. Epoch |
| ---------- | ------- | -----------: | ---------: | ---------: | --------: | -------------: | -----------------: | -----------------: | ----------: |
| Simple CNN | Max     |        98.93 |     0.9892 |     0.0944 |    7.6344 |         230.19 |             0.1858 |              90.70 |          15 |
| Simple CNN | PCA     |    **99.27** | **0.9926** |     0.0944 |    7.6344 |         270.71 |             0.2128 |             890.21 |          14 |
| LeNet      | Max     |    **99.15** | **0.9914** |     0.0617 |    0.4165 |         222.60 |             0.2649 |              38.87 |          15 |
| LeNet      | PCA     |        98.84 |     0.9884 |     0.0617 |    0.4165 |         255.15 |         **0.2055** |             183.48 |           9 |
| ResNet-18  | Max     |        99.52 |     0.9951 |    11.1728 |  150.0902 |         343.60 |             0.2687 |             411.47 |          12 |
| ResNet-18  | PCA     |    **99.59** | **0.9959** |    11.1728 |  150.0902 |         533.71 |             0.2779 |            1995.52 |          14 |

### 5.1 Accuracy
As we know benchmark on MNIST dataset is very high in for deep learning still BCNN and Resnet-18 achived higher accuracy using PCAPool. LeNet
The PCA-based pooling require more GPU memory use, training time and FLOPs for all three architectures. However, convergence speed was higher for BCNN and LeNet, but slightly lower than maxpool for Resnet-18.

## 6. Discussion

The MNIST experiments demonstrate that the effect of PCA-based pooling depends on the underlying CNN architecture.

PCA-based pooling produces a small accuracy improvement in Simple CNN and ResNet-18 but decreases accuracy in LeNet. Therefore, the results do not support the general conclusion that PCA-based pooling is universally better than max pooling.

The computational results provide another important observation. Although the number of trainable parameters remains unchanged, PCA-based pooling substantially increases training time and GPU memory consumption. This is expected because PCA requires additional matrix operations compared with the simple maximum-selection operation used by max pooling.

Therefore, the practical evaluation of PCA-based pooling should consider at least two dimensions:

$$
\text{Representation Quality}
\quad \text{vs.} \quad
\text{Computational Cost}.
$$

A small improvement in accuracy may not necessarily justify a large increase in memory or training cost, particularly for resource-constrained applications.

---

## 7. Limitations

Several limitations should be considered.

First, the current experiment uses only MNIST, which is a relatively simple dataset. Results on more complex datasets such as CIFAR-10, CIFAR-100, and Tiny ImageNet may differ.

Second, the current experiment uses a single random seed. Multiple independent runs are required to determine whether the observed differences are statistically stable.

Third, the current FLOP measurement does not fully represent the computational cost of the PCA operation itself. A custom analytical FLOP calculation or more appropriate profiling method is required.

Finally, convergence should ideally be determined using a validation set rather than the test set. The test set should be reserved for final evaluation.

---

## 8. Future Work

The next stage of the study will extend the same experimental framework to:

$$
\text{MNIST}
\rightarrow
\text{CIFAR-10}
\rightarrow
\text{CIFAR-100}
\rightarrow
\text{Tiny ImageNet}.
$$

Future experiments will also include:

* multiple random seeds;
* confidence intervals or statistical significance analysis;
* detailed PCA computational-cost analysis;
* feature-space visualization;
* analysis of information/variance preservation;
* lightweight CNN architectures;
* comparison with average pooling and learnable projection pooling.

This progression will help determine whether the behavior observed on MNIST generalizes to more challenging visual recognition problems.

---

## References

[1] B. Zhao, X. Dong, Y. Guo, X. Jia, and Y. Huang, “PCA Dimensionality Reduction Method for Image Classification,” *Neural Processing Letters*, vol. 54, pp. 347–368, 2022, doi: 10.1007/s11063-021-10632-5.

[2] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet classification with deep convolutional neural networks,” in *Advances in Neural Information Processing Systems*, vol. 25, 2012.

[3] K. Pearson, “On lines and planes of closest fit to systems of points in space,” *The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science*, vol. 2, no. 11, pp. 559–572, 1901.

[4] H. Hotelling, “Analysis of a complex of statistical variables into principal components,” *Journal of Educational Psychology*, vol. 24, no. 6, pp. 417–441, 1933.

[5] T.-H. Chan, K. Jia, S. Gao, J. Lu, Z. Zeng, and Y. Ma, “PCANet: A simple deep learning baseline for image classification,” *arXiv preprint arXiv:1404.3606*, 2014.

[6] J. Zhou, X. Wu, and others, “Progressive principle component analysis for compressing deep convolutional neural networks,” *Neurocomputing*, vol. 440, pp. 197–206, 2021.

[7] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” *Proceedings of the IEEE*, vol. 86, no. 11, pp. 2278–2324, 1998.

[8] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 770–778, 2016.

[9] A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” in *Advances in Neural Information Processing Systems*, vol. 32, 2019.
