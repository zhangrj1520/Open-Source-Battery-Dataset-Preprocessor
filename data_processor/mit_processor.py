import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from data_processor.base_processor import BaseProcessor


class MitProcessor(BaseProcessor):
    """处理MIT数据集的Processor。"""

    def __init__(self, output_dir="data/processed", mat_path=None):
        super().__init__(output_dir)
        self.mat_path = mat_path
        self._f = None
        self._batch = None
        self._n_cells = 0

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
            I = self._f[cyc_group['I'][ci, 0]][:].flatten()
            V = self._f[cyc_group['V'][ci, 0]][:].flatten()
            t = self._f[cyc_group['t'][ci, 0]][:].flatten()
            T = self._f[cyc_group['T'][ci, 0]][:].flatten()
            Qc = self._f[cyc_group['Qc'][ci, 0]][:].flatten()
            Qd = self._f[cyc_group['Qd'][ci, 0]][:].flatten()

            stages = self._determine_charge_stages(I)

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

    def _determine_charge_stages(self, I):
        """
        仅区分充电/放电 (MIT 1Hz数据无显式静置):
        - I > 0.01A -> stage 1 (充电)
        - I < -0.01A -> stage 3 (放电)
        - |I| <= 0.01A -> 继承前一段的状态 (处理充电中途协议停顿)
        """
        stages = np.zeros(len(I), dtype=int)
        stages[I > 0.01] = 1
        stages[I < -0.01] = 3

        idle = stages == 0
        if idle.any():
            s = pd.Series(stages.astype(float))
            s[idle] = np.nan
            s = s.ffill().fillna(1)
            stages = s.astype(int).values

        return stages

    def _compute_capacity(self, df):
        """统一容量列: 充电阶段用Qc, 放电阶段用Qd, 组内从0开始累计。"""
        df = df.copy()
        df['capacity'] = np.where(
            df['charge_stage'] == 1,
            df['charge_capacity'],
            df['discharge_capacity']
        )
        df['capacity'] = df.groupby(
            ['cycle_number', 'charge_stage']
        )['capacity'].transform(lambda x: x - x.min())
        return df

    def _remove_abnormal_cycles(self, df):
        """异常检测"""
        stages_per_cycle = df.groupby('cycle_number')['charge_stage'].apply(
            lambda x: set(x.unique()))
        valid = stages_per_cycle[stages_per_cycle.apply(
            lambda s: {1, 3}.issubset(s))].index.tolist()
        removed = stages_per_cycle[~stages_per_cycle.index.isin(valid)].index.tolist()

        if removed:
            print("  [MIT] Removed %d cycles missing charge/discharge: %s" % (
                len(removed),
                str([int(c) for c in removed[:10]]) +
                ('...' if len(removed) > 10 else '')))

        return df[df['cycle_number'].isin(valid)].copy()
