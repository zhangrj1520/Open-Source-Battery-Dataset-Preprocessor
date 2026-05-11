"""
MIT电池数据集Processor。

数据结构 (HDF5 v7.3):
- batch/{barcode, channel_id, cycle_life, cycles, policy, policy_readable, summary}
- 每颗电池的cycles是Group, 含 I, V, t, T, Qc, Qd 等字段
- 每个cycle是HDF5 reference, 指向(1, N)的时间序列数据
- 采样率~1Hz, 时间单位: 秒

充电阶段判定:
- 电流 > 0.01A 且 Qc在增加 -> stage 1 (充电)
- 电流 < -0.01A -> stage 3 (放电)
- 电流~0 -> stage 2/4 (静置), 根据前一阶段判定
"""
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from data_processor.base_processor import BaseProcessor


class MitProcessor(BaseProcessor):
    """处理MIT数据集的Processor。

    与XJTU/Tongji不同, MIT的.mat文件包含多颗电池,
    需在load_data中按电池索引逐个处理。
    """

    def __init__(self, output_dir="data/processed", mat_path=None):
        super().__init__(output_dir)
        self.mat_path = mat_path
        self._f = None
        self._batch = None
        self._n_cells = 0
        self._barcodes = []

    def open(self, mat_path):
        """打开.mat文件, 返回电池数量"""
        self.mat_path = Path(mat_path)
        self._f = h5py.File(str(self.mat_path), 'r')
        self._batch = self._f['batch']
        self._n_cells = self._batch['barcode'].shape[0]
        return self._n_cells

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None

    @property
    def n_cells(self):
        return self._n_cells

    def get_cell_name(self, index):
        """返回形如 #1, #2 的编号 (1-based, 按.mat中的行序)"""
        return "#%d" % (index + 1)

    def load_data(self, cell_index):
        """加载单颗电池的全部cycle数据并标准化"""
        if self._f is None:
            raise RuntimeError("Call open() first")

        cyc_ref = self._batch['cycles'][cell_index, 0]
        cyc_group = self._f[cyc_ref]
        n_cycles = cyc_group['I'].shape[0]

        all_cycles = []
        for ci in range(n_cycles):
            # 懒加载每个cycle的数据
            I = self._f[cyc_group['I'][ci, 0]][:].flatten()
            V = self._f[cyc_group['V'][ci, 0]][:].flatten()
            t = self._f[cyc_group['t'][ci, 0]][:].flatten()
            T = self._f[cyc_group['T'][ci, 0]][:].flatten()
            Qc = self._f[cyc_group['Qc'][ci, 0]][:].flatten()
            Qd = self._f[cyc_group['Qd'][ci, 0]][:].flatten()

            # 判定阶段
            stages = self._determine_charge_stages(I, Qc, Qd, t)

            temp_df = pd.DataFrame({
                'cycle_number': ci + 1,
                'time': t - t[0],
                'voltage': V,
                'current': I,
                'temperature': T,
                'charge_capacity': Qc,
                'discharge_capacity': Qd,
                'charge_stage': stages,
            })
            all_cycles.append(temp_df)

        df = pd.concat(all_cycles, ignore_index=True)
        df = self._remove_abnormal_cycles(df)
        df = self._compute_capacity(df)
        return df

    def _determine_charge_stages(self, I, Qc, Qd, t):
        """
        根据电流和容量变化判定阶段:
        - 充电 (1): I > 0.01A
        - 放电 (3): I < -0.01A
        - 静置 (2/4): |I| <= 0.01A, 根据前一active阶段判定
        """
        stages = np.zeros(len(I), dtype=int)
        stages[I > 0.01] = 1
        stages[I < -0.01] = 3

        # 判定静置
        rest = (np.abs(I) <= 0.01)
        if rest.any():
            s = pd.Series(stages.astype(float))
            s[rest] = np.nan
            last_active = s.ffill().fillna(1)
            stages[rest] = np.where(last_active[rest] == 1, 2, 4)

        return stages

    def _compute_capacity(self, df):
        """
        计算累计容量 (Ah)。
        MIT原始数据已有Qc(充电容量)和Qd(放电容量),
        统一为capacity: 充电阶段用Qc, 放电阶段用Qd。
        """
        df = df.copy()
        df['capacity'] = np.where(
            df['charge_stage'].isin([1, 2]),
            df['charge_capacity'],
            df['discharge_capacity']
        )
        # 每个(cluster, stage)组内从0开始累计
        df['capacity'] = df.groupby(
            ['cycle_number', 'charge_stage']
        )['capacity'].transform(lambda x: x - x.min())
        return df

    def _remove_abnormal_cycles(self, df):
        """
        MIT异常cycle检测:
        1. 阶段完整性: 应有charge和discharge
        2. 时间连续性: 相邻数据点间隔 < 600s
        """
        all_cycles = df['cycle_number'].unique()
        valid_mask = pd.Series(True, index=all_cycles)

        # 阶段完整性
        stages_per_cycle = df.groupby('cycle_number')['charge_stage'].apply(
            lambda x: set(x.unique()))
        valid_mask &= stages_per_cycle.apply(lambda s: {1, 3}.issubset(s))

        # 时间连续性
        time_diff = df.groupby('cycle_number')['time'].diff()
        max_gap = time_diff.groupby(df['cycle_number']).max()
        valid_mask &= max_gap < 600

        valid_cycles = valid_mask[valid_mask].index.tolist()
        removed = valid_mask[~valid_mask].index.tolist()

        if removed:
            print("  [MIT] Removed %d abnormal cycles: %s" % (
                len(removed),
                str([int(c) for c in removed[:10]]) +
                ('...' if len(removed) > 10 else '')))

        return df[df['cycle_number'].isin(valid_cycles)].copy()
