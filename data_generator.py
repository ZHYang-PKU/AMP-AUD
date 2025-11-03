import torch
import torch.nn as nn
import numpy as np


class DataGenerator:
    def __init__(self, L, N, M, active_prob=0.1, channel_var=1.0, noise_var=0.1, channel_mode = 'rayleigh',channel_params=None):
        """
        初始化数据生成器
        参数:
            L: 导频长度
            N: 用户总数
            M: 基站天线数
            active_prob: 用户活跃概率（独立同分布）
            channel_var: 信道方差
            noise_var: 噪声方差
        """
        self.L = L
        self.N = N
        self.M = M
        self.active_prob = active_prob
        self.channel_var = channel_var  #高斯信道的功率
        self.noise_var = noise_var     # 接收噪声的功率
        self.channel_params = channel_params or {}
        self.channel_mode = channel_mode

    def _get_channel_params(self):
        """获取信道模式的默认参数"""
        default_params = {
            'rayleigh': {},
            'rician': {'K_factor': 3.0},
            'correlated': {'tx_corr_factor': 0.7, 'rx_corr_factor': 0.5},
            'geometric': {'num_paths': 5, 'max_delay': 10},
            'time_varying': {'time_corr': 0.95}
        }
        return default_params.get(self.channel_mode, {}).copy()

    def generate_pilot_matrix(self, pilot_type='Gaussian'):
        """生成导频矩阵P，维度[L, N], L: pilot length, N: number of users"""
        # 使用正交导频或随机导频
        if pilot_type == 'Gaussian':
            P = torch.randn(self.L, self.N, dtype=torch.complex64)
            P = P/torch.norm(P, dim=0, keepdim=True)
        elif pilot_type == 'zc':
            # Zadoff-Chu序列
            P = torch.zeros((self.L, self.N), dtype=torch.complex64)
            # 选择一个与序列长度互质的根索引
            root_index = 1
            for n in range(self.N):
                # 为每个用户/天线分配不同的根索引
                u = (root_index + n) % self.L
                if u == 0:
                    u = 1  # 避免u=0
                # 生成ZC序列
                k = torch.arange(self.L)
                if self.L % 2 == 0:  # 偶数长度
                    zc_seq = torch.exp(-1j * torch.pi * u * k * (k + 1) / self.L)
                else:  # 奇数长度
                    zc_seq = torch.exp(-1j * torch.pi * u * k * k / self.L)
                P[:, n] = zc_seq
            # 归一化
            P = P / torch.norm(P, dim=0, keepdim=True)
        elif pilot_type == 'dft':
            # DFT矩阵导频
            # 生成L点DFT矩阵
            dft_matrix = torch.fft.fft(torch.eye(self.L))
            # 如果N <= L，选择不同的列
            if self.N <= self.L:
                # 均匀选择列
                col_indices = torch.linspace(0, self.L - 1, self.N, dtype=torch.long)
                P = dft_matrix[:, col_indices]
            else:
                # 如果N > L，重复使用DFT矩阵的列
                repeats = (self.N + self.L - 1) // self.L
                P = dft_matrix.repeat(1, repeats)
                P = P[:, :self.N]
            # 归一化
            P = P / torch.norm(P, dim=0, keepdim=True)
        else:
            raise ValueError(f"pilot type {pilot_type} not supported.")

        return P   # [L, N]

    def generate_active_status(self, batch_size, distribution_mode=0):
        """
        生成活跃状态A，维度[batch_size, N]
        参数:
        batch_size: 批次大小
        distribution_mode: 分布模式
            0 - 独立同分布（伯努利）
            1 - 独立不同分布（每个用户不同概率）
            2 - 分组相关分布（组内用户相关）
            3 - 时间相关分布（批次内时间相关性）
            4 - 社区结构分布（社区内相关性）
        """
        if distribution_mode == 0:
            # 模式0：独立同分布，伯努利分布
            A = torch.bernoulli(torch.ones(batch_size, self.N) * self.active_prob)

        elif distribution_mode == 1:
            # 模式1：独立不同分布，每个用户有不同的活跃概率
            # 生成每个用户的基础活跃概率（在0.1到0.9之间）
            user_probs = torch.rand(self.N) * 0.8 + 0.1

            # 为每个批次样本生成活跃状态
            A = torch.bernoulli(user_probs.unsqueeze(0).repeat(batch_size, 1))

        elif distribution_mode == 2:
            # 模式2：分组相关分布
            # 将用户随机分为若干组
            num_groups = max(2, self.N // 5)  # 分组数量
            group_assignments = torch.randint(0, num_groups, (self.N,))

            # 为每个组生成基础活跃概率
            group_probs = torch.rand(num_groups) * 0.7 + 0.1  # 在0.1-0.8之间

            # 为每个批次生成组活跃状态
            group_active = torch.bernoulli(
                group_probs.unsqueeze(0).repeat(batch_size, 1)
            )

            # 为每个用户分配基于组的活跃概率，添加一些随机变化
            user_probs = torch.zeros((batch_size, self.N))
            for i in range(self.N):
                group_idx = group_assignments[i]
                base_prob = group_probs[group_idx]
                # 添加用户特定的随机变化 (±0.1)
                user_specific_var = (torch.rand(batch_size) - 0.5) * 0.2
                user_probs[:, i] = torch.clamp(base_prob + user_specific_var, 0.05, 0.95)

            # 生成用户活跃状态
            A = torch.bernoulli(user_probs)

        elif distribution_mode == 3:
            # 模式3：时间相关分布（假设批次内存在时间顺序）
            A = torch.zeros((batch_size, self.N))

            # 生成初始状态
            initial_state = torch.bernoulli(torch.full((self.N,), self.active_prob))
            A[0] = initial_state

            # 为每个时间步生成状态
            for i in range(1, batch_size):
                # 每个用户有transition_prob概率保持与前一个样本相同的状态
                transition_prob = 0.7  # 状态转移概率

                keep_previous = torch.bernoulli(torch.full((self.N,), transition_prob))
                # 1-transition_prob概率随机改变状态
                random_change = torch.bernoulli(torch.full((self.N,), self.active_prob))
                A[i] = keep_previous * A[i - 1] + (1 - keep_previous) * random_change

        elif distribution_mode == 4:
            # 模式4：社区结构分布
            # 创建社区结构（小世界网络或随机块模型）
            num_communities = max(2, self.N // 10)  # 社区数量
            community_assignments = torch.randint(0, num_communities, (self.N,))

            # 生成社区间的连接概率矩阵
            intra_community_prob = 0.8  # 社区内高连接概率
            inter_community_prob = 0.2  # 社区间低连接概率

            # 生成社区活跃概率
            community_probs = torch.rand(num_communities) * 0.6 + 0.2  # 0.2-0.8

            # 为每个批次生成活跃状态
            A = torch.zeros((batch_size, self.N))

            for b in range(batch_size):
                # 为每个社区生成基础活跃状态
                community_active = torch.bernoulli(community_probs)

                # 为每个用户分配活跃状态，考虑社区结构
                for i in range(self.N):
                    comm_idx = community_assignments[i]
                    base_prob = community_probs[comm_idx]

                    # 考虑邻居影响（同一社区的其他用户）
                    same_community = (community_assignments == comm_idx)
                    same_community[i] = False  # 排除自己

                    if same_community.sum() > 0:
                        # 邻居的活跃状态会影响当前用户
                        neighbor_influence = torch.sum(A[b, same_community]) / same_community.sum()
                        influence_factor = 0.3  # 邻居影响因子
                        adjusted_prob = base_prob * (1 - influence_factor) + neighbor_influence * influence_factor
                        adjusted_prob = torch.clamp(adjusted_prob, 0.05, 0.95)
                    else:
                        adjusted_prob = base_prob

                    # 添加随机噪声
                    noise = (torch.rand(1) - 0.5) * 0.1
                    final_prob = torch.clamp(adjusted_prob + noise, 0.05, 0.95)

                    A[b, i] = torch.bernoulli(final_prob)

        else:
            raise ValueError(f"Unsupported activity distribution mode: {distribution_mode}")

        return A


    def generate_channel(self, batch_size, **override_params):
        """生成信道矩阵H，维度[batch_size, N, M]"""
        # TODO:生成更复杂的信道分布
        """
            生成信道矩阵H，维度[batch_size, N, M]

            参数:
                batch_size: 批次大小
                channel_mode: 信道模式
                    'rayleigh' - 瑞利衰落信道
                    'rician' - 莱斯衰落信道
                    'correlated' - 空间相关MIMO信道
                    'geometric' - 几何多径信道
                    'time_varying' - 时变信道
                channel_params: 信道参数字典
            """
        # 合并参数：实例参数 < 类默认参数 < 调用时参数
        params = self._get_channel_params()
        params.update(self.channel_params)  # 实例级别的参数
        params.update(override_params)  # 调用时参数（优先级最高）

        if self.channel_mode == 'rayleigh':
            # 瑞利衰落信道：独立同分布复高斯
            H_real = torch.randn(batch_size, self.N, self.M) * np.sqrt(self.channel_var / 2)
            H_imag = torch.randn(batch_size, self.N, self.M) * np.sqrt(self.channel_var / 2)
            H = torch.complex(H_real, H_imag)

        elif self.channel_mode == 'rician':
            # 莱斯衰落信道：存在主导路径
            K = params.get('K_factor', 3.0)  # K因子，主导路径与散射路径功率比

            # 主导路径分量（确定性）
            los_component = torch.ones(batch_size, self.N, self.M, dtype=torch.complex64)
            los_power = K / (K + 1)

            # 散射路径分量（随机）
            H_real = torch.randn(batch_size, self.N, self.M) * np.sqrt(self.channel_var / (2 * (K + 1)))
            H_imag = torch.randn(batch_size, self.N, self.M) * np.sqrt(self.channel_var / (2 * (K + 1)))
            nlos_component = torch.complex(H_real, H_imag)

            # 组合主导路径和散射路径
            H = np.sqrt(los_power) * los_component + nlos_component

        elif self.channel_mode == 'correlated':
            # 空间相关MIMO信道
            # 生成发送端和接收端相关矩阵
            tx_corr = self._generate_correlation_matrix(self.N, params.get('tx_corr_factor', 0.7))
            rx_corr = self._generate_correlation_matrix(self.M, params.get('rx_corr_factor', 0.5))

            # 生成独立同分布信道
            H_iid_real = torch.randn(batch_size, self.N, self.M) * np.sqrt(self.channel_var / 2)
            H_iid_imag = torch.randn(batch_size, self.N, self.M) * np.sqrt(self.channel_var / 2)
            H_iid = torch.complex(H_iid_real, H_iid_imag)

            # 应用空间相关性
            tx_corr_sqrt = torch.linalg.cholesky(tx_corr)
            rx_corr_sqrt = torch.linalg.cholesky(rx_corr)

            H = torch.matmul(tx_corr_sqrt, torch.matmul(H_iid, rx_corr_sqrt))

        elif self.channel_mode == 'geometric':
            # 几何多径信道
            num_paths = params.get('num_paths', 5)  # 多径数量
            max_delay = params.get('max_delay', 10)  # 最大时延

            H = torch.zeros((batch_size, self.N, self.M), dtype=torch.complex64)

            for b in range(batch_size):
                # 为每个样本生成多径参数
                path_gains = torch.randn(num_paths)  # 路径增益
                path_delays = torch.rand(num_paths) * max_delay  # 路径时延
                path_aoa = torch.rand(num_paths) * 2 * np.pi  # 到达角
                path_aod = torch.rand(num_paths) * 2 * np.pi  # 出发角

                # 假设均匀线性阵列
                d = 0.5  # 天线间距（波长的一半）

                # 发送端阵列响应向量
                tx_array = torch.arange(0, self.N)
                tx_response = torch.exp(1j * 2 * np.pi * d * tx_array.unsqueeze(1) * torch.sin(path_aod))

                # 接收端阵列响应向量
                rx_array = torch.arange(0, self.M)
                rx_response = torch.exp(1j * 2 * np.pi * d * rx_array.unsqueeze(1) * torch.sin(path_aoa))

                # 构建信道矩阵
                for p in range(num_paths):
                    path_matrix = torch.outer(tx_response[:, p], rx_response[:, p])
                    H[b] += path_gains[p] * torch.exp(-1j * 2 * np.pi * path_delays[p]) * path_matrix

                # 功率归一化
                H_power = torch.mean(torch.abs(H[b]) ** 2)
                H[b] = H[b] / torch.sqrt(H_power) * np.sqrt(self.channel_var)

        elif self.channel_mode == 'time_varying':
            # 时变信道（批次内时间相关性）
            # 假设批次中的样本是连续时间点
            rho = params.get('time_corr', 0.95)  # 时间相关系数

            H = torch.zeros((batch_size, self.N, self.M), dtype=torch.complex64)

            # 生成初始信道
            H_prev_real = torch.randn(self.N, self.M) * np.sqrt(self.channel_var / 2)
            H_prev_imag = torch.randn(self.N, self.M) * np.sqrt(self.channel_var / 2)
            H_prev = torch.complex(H_prev_real, H_prev_imag)
            H[0] = H_prev

            # 生成后续时间点的信道
            for t in range(1, batch_size):
                # 新信道 = rho * 旧信道 + sqrt(1-rho^2) * 新噪声
                innovation_real = torch.randn(self.N, self.M) * np.sqrt(self.channel_var / 2)
                innovation_imag = torch.randn(self.N, self.M) * np.sqrt(self.channel_var / 2)
                innovation = torch.complex(innovation_real, innovation_imag)

                H[t] = rho * H_prev + np.sqrt(1 - rho ** 2) * innovation
                H_prev = H[t]

        else:
            raise ValueError(f"Unsupported channel mode: {self.channel_mode}")

        return H

    def _generate_correlation_matrix(self, size, correlation_factor):
        """生成指数衰减的相关矩阵"""
        corr_matrix = torch.zeros(size, size)
        for i in range(size):
            for j in range(size):
                corr_matrix[i, j] = correlation_factor ** abs(i - j)
        return corr_matrix




    def generate_observation(self, P, A, H):
        """生成观测信号Y = P A H + Z"""
        batch_size = A.shape[0]

        # 构造对角矩阵A
        A_diag = torch.diag_embed(A.to(torch.complex64))

        # 计算PAH
        PA = torch.matmul(P, A_diag)  # [batch_size, L, N]
        PAH = torch.matmul(PA, H)  # [batch_size, L, M]

        # 生成噪声
        Z_real = torch.randn(batch_size, self.L, self.M) * np.sqrt(self.noise_var / 2)
        Z_imag = torch.randn(batch_size, self.L, self.M) * np.sqrt(self.noise_var / 2)
        Z = torch.complex(Z_real, Z_imag)

        # 生成观测
        Y = PAH + Z

        return Y

    def generate_batch(self, batch_size):
        """生成一个批次的数据"""
        P = self.generate_pilot_matrix()
        A = self.generate_active_status(batch_size)
        H = self.generate_channel(batch_size)
        Y = self.generate_observation(P, A, H)

        return Y, A, H, P

    def generate_annealed_data(self, batch_size, sigma_t):
        """生成退火训练数据"""
        P = self.generate_pilot_matrix()
        A_true = self.generate_active_status(batch_size)

        # 添加高斯噪声生成退火版本
        A_tilde = A_true + sigma_t * torch.randn_like(A_true)

        H = self.generate_channel(batch_size)
        Y = self.generate_observation(P, A_true, H)

        return Y, A_tilde, A_true, H, P


if __name__ == '__main__':

    L = 32
    N = 64
    M = 128
    active_prob = 0.1
    channel_var = 1.0
    noise_var = 0.1
    batch_size = 16
    sigma_t = 0.1

    data_generator = DataGenerator(L, N, M, active_prob, channel_var, noise_var, channel_mode='time_varying')
    # 生成单个样本
    P = data_generator.generate_pilot_matrix(pilot_type='dft')
    A = data_generator.generate_active_status(1, distribution_mode=2)
    #H = data_generator.generate_channel(1)
    #H = data_generator.generate_channel(1,  K_factor = 3.0)
    #H = data_generator.generate_channel(1,  num_paths = 5, max_delay =10 )
    H = data_generator.generate_channel(1,  time_corr=0.95)
    Y = data_generator.generate_observation(P, A, H)

    # 生成一个batch的样本（不退火）
    #Y_batch, A_batch, H_batch, P_batch = data_generator.generate_batch(batch_size)

    # 生成一个batch的样本（退火）
    Y_batch, A_batch_tilde, A_batch, H_batch, P_batch = data_generator.generate_annealed_data(batch_size, sigma_t)



    print(f'shape of A_batch: {A_batch.shape}')
    print(f'shape of A_batch_tilde: {A_batch_tilde.shape}')


    print(f'shape of H: {H.shape}')
    print(f'shape of H_batch: {H_batch.shape}')
    print(f'shape of P: {P.shape}')
    print(f'shape of P_batch: {P_batch.shape}')
    #print(f'shape of A_tilde: {A_tilde.shape}')