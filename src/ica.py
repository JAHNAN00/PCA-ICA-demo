import numpy as np
import matplotlib.pyplot as plt

class ICA:
    def __init__(self, n_components: int, max_iter: int = 2000, tol: float = 1e-10):
        """
        初始化 ICA 类
        :param n_components: 要提取的独立分量数量
        :param max_iter: 最大迭代次数
        :param tol: 收敛阈值
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def _whiten(self, X):
        """
        白化处理
        :param X: 输入信号矩阵 (n_samples, n_features)
        :return: 白化后的矩阵
        """
        X -= np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        D = np.diag(1.0 / np.sqrt(eigenvalues))
        whitening_matrix = eigenvectors @ D @ eigenvectors.T
        return X @ whitening_matrix

    def fit_transform(self, X):
        """
        对输入矩阵进行 ICA 并返回分离出的独立分量
        :param X: 输入信号矩阵 (n_samples, n_features)
        :return: 分离出的独立分量矩阵 (n_samples, n_components)
        """
        X = self._whiten(X)
        n_samples, n_features = X.shape
        W = np.random.randn(self.n_components, n_features)

        for _ in range(self.max_iter):
            W_new = np.zeros_like(W)
            for i in range(self.n_components):
                w = W[i, :]
                w = np.mean(X * (X @ w).reshape(-1, 1), axis=0) - np.mean(1 - (X @ w) ** 2) * w
                for j in range(i):
                    w -= np.dot(w, W_new[j]) * W_new[j]
                W_new[i] = w / np.linalg.norm(w)
            
            if np.max(np.abs(np.abs(np.diag(W_new @ W.T)) - 1)) < self.tol:
                break
            W = W_new
        
        S = X @ W.T
        return S


if __name__ == "__main__":
    # 生成两个不同的正弦信号
    np.random.seed(0)
    t = np.linspace(0, 1, 1000)
    s1 = np.sin(2 * np.pi * 5 * t)  # 频率为5Hz的正弦信号
    s2 = np.sin(2 * np.pi * 10 * t + np.pi / 4)  # 频率为10Hz，相位偏移π/4的正弦信号

    # 混合信号
    S = np.vstack([s1, s2]).T
    A = np.array([[1, 0.5], [0.5, 1]])  # 混合矩阵
    X = S @ A.T  # 生成混合信号
    print(np.shape(S))
    print(np.shape(X))

    # 使用 ICA 分离信号
    ica = ICA(n_components=2)
    S_estimated = ica.fit_transform(X)

    # 绘图
    fig, axes = plt.subplots(5, 1, figsize=(8, 10))
    axes[0].plot(t, s1)
    axes[0].set_title("信号 1")
    axes[1].plot(t, s2)
    axes[1].set_title("信号 2")
    axes[2].plot(t, X[:, 0], label="混合信号 1")
    axes[2].plot(t, X[:, 1], label="混合信号 2")
    axes[2].set_title("混合信号")
    axes[2].legend()
    axes[3].plot(t, S_estimated[:, 0])
    axes[3].set_title("分离出的信号 1")
    axes[4].plot(t, S_estimated[:, 1])
    axes[4].set_title("分离出的信号 2")
    plt.tight_layout()
    plt.savefig('ica.png')
