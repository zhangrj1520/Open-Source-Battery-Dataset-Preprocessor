import numpy as np
import pandas as pd
from scipy.io import loadmat
from pathlib import Path
from data_processor.base_processor import BaseProcessor


class NasaProcessor(BaseProcessor):
    """处理NASA电池数据集的Processor"""

    def load_data(self, file_path):
        """加载NASA .mat数据并执行标准化转换"""
        mat = loadmat(str(file_path))
        cell_name = Path(file_path).stem
        cell = mat[cell_name][0, 0]
        cycles_raw = cell['cycle'][0]

        all_cycles = []
        cycle_num = 0

        for i in range(len(cycles_raw)):
            c = cycles_raw[i]
            op_type = str(c['type'][0])

            if op_type == 'impedance':
                continue  # 过滤EIS测量
            if op_type not in ('charge', 'discharge'):
                continue

            d = c['data'][0, 0]
            n_rows = len(d['Time'].flatten())

            if op_type == 'charge' and n_rows > 0:
                cycle_num += 1

            stages = self._determine_charge_stages(op_type, n_rows)
            temp_df = pd.DataFrame({
                'cycle_number': cycle_num,
                'type': op_type,
                'time': d['Time'].flatten() - d['Time'].flatten()[0],
                'voltage': d['Voltage_measured'].flatten(),
                'current': d['Current_measured'].flatten(),
                'temperature': d['Temperature_measured'].flatten(),
                'charge_stage': stages,
            })
            all_cycles.append(temp_df)

        if not all_cycles:
            raise ValueError(f"No charge/discharge data found in {file_path}")

        df = pd.concat(all_cycles, ignore_index=True)
        df = self._remove_abnormal_cycles(df)
        df = self._compute_capacity(df)
        return df

    def _determine_charge_stages(self, op_type, n_rows):
        """
        根据操作类型直接判定阶段:
        - charge → 1
        - discharge → 3
        无静置阶段。
        """
        stage = 1 if op_type == 'charge' else 3
        return np.full(n_rows, stage, dtype=int)

    def _compute_capacity(self, df):
        """
        计算容量 (Ah)。
        NASA数据charge阶段无显式Capacity字段, 通过安时积分积分计算。
        每个(cycle_number, charge_stage)组内独立累计。
        """
        df = df.copy()
        dt = df.groupby(['cycle_number', 'charge_stage'])['time'].diff().fillna(0)
        group_key = df['cycle_number'].astype(str) + '_' + df['charge_stage'].astype(str)
        df['capacity'] = (df['current'].abs() * dt).groupby(group_key).cumsum() / 3600.0
        return df

    def _remove_abnormal_cycles(self, df):
        """异常检测"""
        all_cycles = df['cycle_number'].unique()
        valid_mask = pd.Series(True, index=all_cycles)

        # 阶段完整
        stages_per_cycle = df.groupby('cycle_number')['charge_stage'].apply(
            lambda x: set(x.unique()))
        valid_mask &= stages_per_cycle.apply(lambda s: {1, 3}.issubset(s))

        # 时间连续
        time_diff = df.groupby('cycle_number')['time'].diff()
        max_gap = time_diff.groupby(df['cycle_number']).max()
        valid_mask &= max_gap < 600

        valid_cycles = valid_mask[valid_mask].index.tolist()

        return df[df['cycle_number'].isin(valid_cycles)].copy()
