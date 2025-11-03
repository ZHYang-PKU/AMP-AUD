import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
from scipy.special import gamma, gammainc
import matplotlib.pyplot as plt
from data_generator import DataGenerator


class AMPDetector:
    """基于AMP的联合活跃用户检测和信道估计器"""

    def __init__(self, L, N, M, active_prob=0.1, channel_var=1.0, noise_var=0.1,
                 max_iter=50, tol=1e-6, device='cuda'):
        """
        初始化AMP检测器

        参数:
            L: 导频长度
            N: 用户总数
            M: 基站天线数
            active_prob: 用户活跃概率
            channel_var: 信道方差
            noise_var: 噪声方差
            max_iter: 最大迭代次数
            tol: 收敛容忍度
            device: 计算设备
        """
        self.L = L
        self.N = N
        self.M = M
        self.active_prob = active_prob
        self.channel_var = channel_var
        self.noise_var = noise_var
        self.max_iter = max_iter
        self.tol = tol
        self.device = device

        # 状态演化参数
        self.tau_t = torch.tensor(np.sqrt(noise_var), device=device, dtype=torch.float32)

    def _compute_denoiser(self, R_t, tau_t, g_n=None):
        """
        计算MMSE去噪器 (论文公式52)

        参数:
            R_t: 匹配滤波输出 [batch_size, N, M] 或 [N, M]
            tau_t: 噪声水平
            g_n: 大尺度衰落系数 [N] 或标量

        返回:
            X_est: 去噪后的信号估计
            div: Onsager项需要的散度
        """
        if not R_t.is_complex():
            R_t = R_t.to(torch.complex64)

        if len(R_t.shape) == 3:
            batch_size, N, M = R_t.shape
            R_norm = torch.norm(R_t, dim=2, keepdim=True) ** 2  # [batch_size, N, 1]
        else:
            N, M = R_t.shape
            R_norm = torch.norm(R_t, dim=1, keepdim=True) ** 2  # [N, 1]

        # 如果没有提供g_n，使用默认值
        if g_n is None:
            g_n = torch.tensor(np.sqrt(self.channel_var), device=R_t.device, dtype=torch.float32)

        # 确保g_n有正确的形状
        if isinstance(g_n, (int, float)):
            g_n = torch.tensor(g_n, device=R_t.device, dtype=torch.float32)
        if g_n.dim() == 0:
            g_n = g_n.unsqueeze(0).repeat(N)
        g_n = g_n.float()

        # 重塑g_n以匹配R_t的形状
        if len(R_t.shape) == 3:
            g_n = g_n.unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
        else:
            g_n = g_n.unsqueeze(-1)  # [N, 1]
        # 确保tau_t是浮点数类型
        tau_t = tau_t.float()

        # 计算Delta (论文公式20)
        Delta = 1 / tau_t ** 2 - 1 / (g_n ** 2 + tau_t ** 2)

        # 计算分母项
        denominator = 1 + ((1 - self.active_prob) / self.active_prob) * \
                      ((g_n ** 2 + tau_t ** 2) / tau_t ** 2) ** self.M * \
                      torch.exp(-Delta * R_norm)

        # 计算MMSE估计 (论文公式52)
        scaling_factor = (g_n ** 2 / (g_n ** 2 + tau_t ** 2)) / denominator
        X_est = scaling_factor * R_t

        # 计算导数用于Onsager项
        # 这里简化处理，使用数值近似
        if len(R_t.shape) == 3:
            div = torch.mean(scaling_factor, dim=(1, 2))  # [batch_size]
        else:
            div = torch.mean(scaling_factor)  # 标量

        return X_est, div

    def _update_tau(self, Z_t, L, M):
        """更新噪声水平估计"""
        return torch.norm(Z_t, dim=(1, 2)) / torch.sqrt(torch.tensor(L * M, device=Z_t.device, dtype=torch.float32))

    def detect(self, Y, P, g_n=None, return_history=False):
        """
        AMP算法进行联合检测和估计

        参数:
            Y: 接收信号 [batch_size, L, M] 或 [L, M]
            P: 导频矩阵 [L, N]
            g_n: 大尺度衰落系数 [N] 或标量
            return_history: 是否返回迭代历史

        返回:
            A_est: 活跃状态估计 [batch_size, N] 或 [N]
            H_est: 信道估计 [batch_size, N, M] 或 [N, M]
            history: 迭代历史 (如果return_history=True)
        """
        # 确保输入为复数类型
        Y = Y.to(torch.complex64)
        P = P.to(torch.complex64)

        # 处理批量维度
        if len(Y.shape) == 3:
            batch_size, L, M = Y.shape
            single_batch = False
        else:
            batch_size = 1
            L, M = Y.shape
            Y = Y.unsqueeze(0)  # [1, L, M]
            single_batch = True

        # 初始化
        X_t = torch.zeros((batch_size, self.N, self.M), dtype=torch.complex64, device=self.device)
        Z_t = Y.clone()  # [batch_size, L, M]

        # 如果g_n是标量，扩展到所有用户
        if g_n is not None:
            if g_n.dim() == 0:
                g_n = g_n * torch.ones(self.N, device=self.device, dtype=torch.float32)
            else:
                g_n = g_n.to(torch.float32)

        # 迭代历史
        if return_history:
            history = {
                'X': [],
                'Z': [],
                'tau': []
            }

        # AMP迭代
        for t in range(self.max_iter):
            # 匹配滤波
            R_t = torch.matmul(P.conj().T, Z_t) + X_t  # [batch_size, N, M]

            # MMSE去噪
            X_t_plus_1, div = self._compute_denoiser(R_t, self.tau_t, g_n)

            # 更新残差 (包含Onsager项)
            # 计算 PA = P @ X_t_plus_1
            print(f"Shape of P: {P.shape}")
            print(f"Shape of X_t_plus_1: {X_t_plus_1.shape}")
            PA = torch.matmul(P, X_t_plus_1.to(torch.complex64))  # [batch_size, L, M]
            Z_t_plus_1 = Y - PA

            # 添加Onsager项
            if len(div.shape) == 0:  # 单样本情况
                onsager_term = (self.N / self.L) * div * Z_t
            else:  # 批量情况
                onsager_term = (self.N / self.L) * div.view(-1, 1, 1) * Z_t

            Z_t_plus_1 = Z_t_plus_1 + onsager_term

            # 更新噪声水平估计
            tau_t_plus_1 = self._update_tau(Z_t_plus_1, self.L, self.M)

            # 检查收敛
            if t > 0:
                diff = torch.norm(X_t_plus_1 - X_t, dim=(1, 2)) / torch.norm(X_t, dim=(1, 2))
                if torch.max(diff) < self.tol:
                    break

            # 更新变量
            X_t = X_t_plus_1
            Z_t = Z_t_plus_1
            self.tau_t = torch.mean(tau_t_plus_1)  # 使用平均值作为全局tau

            # 保存历史
            if return_history:
                history['X'].append(X_t.detach().cpu())
                history['Z'].append(Z_t.detach().cpu())
                history['tau'].append(self.tau_t.detach().cpu())

        # 活跃用户检测
        X_norm = torch.norm(X_t, dim=2)  # [batch_size, N]

        # 使用基于能量阈值的检测
        # 阈值可以根据经验设置或通过理论分析得到
        threshold = 0.1 * torch.max(X_norm, dim=1, keepdim=True)[0]
        A_est = (X_norm > threshold).float()

        # 信道估计就是X_t本身
        H_est = X_t

        if single_batch:
            A_est = A_est.squeeze(0)
            H_est = H_est.squeeze(0)

        if return_history:
            return A_est, H_est, history
        else:
            return A_est, H_est

    def evaluate_performance(self, A_true, A_est, H_true, H_est):
        """
        评估检测和估计性能

        参数:
            A_true: 真实活跃状态 [batch_size, N]
            A_est: 估计活跃状态 [batch_size, N]
            H_true: 真实信道 [batch_size, N, M]
            H_est: 估计信道 [batch_size, N, M]

        返回:
            metrics: 性能指标字典
        """
        # 确保是批量形式
        if len(A_true.shape) == 1:
            A_true = A_true.unsqueeze(0)
            A_est = A_est.unsqueeze(0)
            H_true = H_true.unsqueeze(0)
            H_est = H_est.unsqueeze(0)

        batch_size = A_true.shape[0]

        # 计算检测性能
        tp = torch.sum((A_est == 1) & (A_true == 1), dim=1)  # 真阳性
        fp = torch.sum((A_est == 1) & (A_true == 0), dim=1)  # 假阳性
        fn = torch.sum((A_est == 0) & (A_true == 1), dim=1)  # 假阴性
        tn = torch.sum((A_est == 0) & (A_true == 0), dim=1)  # 真阴性

        # 避免除零
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1_score = 2 * precision * recall / (precision + recall + 1e-10)

        # 计算信道估计性能
        mse_channel = torch.mean(torch.abs(H_est - H_true) ** 2, dim=(1, 2))
        nmse_channel = mse_channel / torch.mean(torch.abs(H_true) ** 2, dim=(1, 2))

        # 只对活跃用户计算信道估计误差
        active_mask = A_true.unsqueeze(-1)  # [batch_size, N, 1]
        H_true_active = H_true * active_mask
        H_est_active = H_est * active_mask

        active_mse = torch.sum(torch.abs(H_est_active - H_true_active) ** 2, dim=(1, 2)) / \
                     (torch.sum(active_mask, dim=(1, 2)) + 1e-10)
        active_nmse = active_mse / (torch.sum(torch.abs(H_true_active) ** 2, dim=(1, 2)) /
                                    (torch.sum(active_mask, dim=(1, 2)) + 1e-10) + 1e-10)

        metrics = {
            'precision': precision.mean().item(),
            'recall': recall.mean().item(),
            'f1_score': f1_score.mean().item(),
            'mse_channel': mse_channel.mean().item(),
            'nmse_channel': nmse_channel.mean().item(),
            'active_mse': active_mse.mean().item(),
            'active_nmse': active_nmse.mean().item()
        }

        return metrics


# 使用示例
def evaluate_amp_detector():
    """测试AMP检测器"""
    # 参数设置
    L = 64  # 导频长度
    N = 100  # 用户总数
    M = 8  # 基站天线数
    batch_size = 10
    active_prob = 0.1
    channel_var = 1.0
    noise_var = 0.01

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # 生成数据
    data_gen = DataGenerator(L, N, M, active_prob, channel_var, noise_var)
    P = data_gen.generate_pilot_matrix('Gaussian')
    A_true = data_gen.generate_active_status(batch_size)
    H_true = data_gen.generate_channel(batch_size)
    #print(f"Device of P: {P.device}")
    #print(f"Device of A_true: {A_true.device}")
    #print(f"Device of H_true: {H_true.device}")
    Y = data_gen.generate_observation(P, A_true, H_true)
    P=P.to(device)
    A_true = A_true.to(device)
    H_true = H_true.to(device)
    Y = Y.to(device)

    # 创建AMP检测器
    detector = AMPDetector(L, N, M, active_prob, channel_var, noise_var, device=device)

    # 执行检测
    A_est, H_est = detector.detect(Y, P)

    # 评估性能
    metrics = detector.evaluate_performance(A_true, A_est, H_true, H_est)

    print("AMP检测器性能:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # 可视化结果
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 真实活跃状态
    axes[0, 0].imshow(A_true.cpu().numpy(), aspect='auto', cmap='Blues')
    axes[0, 0].set_title('True Activity Pattern')
    axes[0, 0].set_xlabel('Users')
    axes[0, 0].set_ylabel('Samples')

    # 估计活跃状态
    axes[0, 1].imshow(A_est.cpu().numpy(), aspect='auto', cmap='Blues')
    axes[0, 1].set_title('Estimated Activity Pattern')
    axes[0, 1].set_xlabel('Users')
    axes[0, 1].set_ylabel('Samples')

    # 信道估计误差
    channel_error = torch.abs(H_est - H_true).mean(dim=2).cpu().numpy()
    im = axes[1, 0].imshow(channel_error, aspect='auto', cmap='hot')
    axes[1, 0].set_title('Channel Estimation Error')
    axes[1, 0].set_xlabel('Users')
    axes[1, 0].set_ylabel('Samples')
    plt.colorbar(im, ax=axes[1, 0])

    # 性能指标
    metrics_text = '\n'.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
    axes[1, 1].text(0.1, 0.9, metrics_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top')
    axes[1, 1].set_title('Performance Metrics')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

    return metrics


if __name__ == "__main__":
    evaluate_amp_detector()