import pickle
import numpy as np
import pandas as pd
from data_processor.base_processor import BaseProcessor


class HustProcessor(BaseProcessor):
    """ 处理HUST数据集的Processor """

    def load_data(self, file_path):
        """ 加载 HUST .pkl 数据并执行标准化转换 """
        with open(file_path, 'rb') as f:
            raw = pickle.load(f)

        cell_name = list(raw.keys())[0]
        cell_data = raw[cell_name]

        all_cycles = []
        for cyc_num, df_cyc in cell_data['data'].items():
            if len(df_cyc) == 0:
                continue
            
            # 判定充放电状态
            stages = self._determine_charge_stages(df_cyc['Status'])

            temp_df = pd.DataFrame({
                'cycle_number': cyc_num,
                'time': df_cyc['Time (s)'].values,
                'voltage': df_cyc['Voltage (V)'].values,
                'current': df_cyc['Current (mA)'].values / 1000.0,
                'capacity': df_cyc['Capacity (mAh)'].values / 1000.0,
                'charge_stage': stages,
            })

            # 每个cycle的时间从0开始
            temp_df['time'] = temp_df['time'] - temp_df['time'].iloc[0]
            all_cycles.append(temp_df)

        df = pd.concat(all_cycles, ignore_index=True)

        # 异常cycle剔除
        df = self._remove_abnormal_cycles(df)

        return df


    def _determine_charge_stages(self, status_series):
        """
        利用Status列直接划分。
        - 1: 充电 (Status包含 'charge' 但不包含 'discharge')
        - 3: 放电 (Status包含 'discharge')
        """
        status_str = status_series.astype(str)
        is_discharge = status_str.str.contains('discharge', case=False)
        stages = np.where(is_discharge, 3, 1)
        return stages


    def _remove_abnormal_cycles(self, df):
        """ 时间连续 """
        time_diff = df.groupby('cycle_number')['time'].diff()
        max_gap_per_cycle = time_diff.groupby(df['cycle_number']).max()
        time_valid = max_gap_per_cycle < 600

        valid_cycles = time_valid[time_valid].index.tolist()

        return df[df['cycle_number'].isin(valid_cycles)].copy()
