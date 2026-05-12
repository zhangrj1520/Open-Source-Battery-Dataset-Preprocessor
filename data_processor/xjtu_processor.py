from scipy.io import loadmat
import pandas as pd
import numpy as np
from data_processor.base_processor import BaseProcessor


class XjtuProcessor(BaseProcessor):
    """处理XJTU数据集的Processor"""
    def load_data(self, file_path):

        mat_data = loadmat(file_path)
        data_array = mat_data['data']
        
        if data_array.shape[0] == 1:
            data_array = data_array.squeeze()
        
        all_cycles = []
        
        for i in range(len(data_array)):
            cycle_struct = data_array[i]
            current = np.ravel(cycle_struct['current_A'])
            
            # 获取时间戳
            raw_times = cycle_struct['system_time'].flatten()
            time_strings = [str(t.item()) if hasattr(t, 'item') else str(t) for t in raw_times]
            t_series = pd.to_datetime(time_strings, format='%Y-%m-%d,%H:%M:%S')
            relative_seconds = (t_series - t_series[0]).total_seconds()
            
            # 判定充放电状态
            stages = self._determine_charge_stages(current)

            # 创建DataFrame
            temp_df = pd.DataFrame({
                'cycle_number': i + 1,
                'time': relative_seconds,
                'voltage': np.ravel(cycle_struct['voltage_V']),
                'current': current,
                'temperature': np.ravel(cycle_struct['temperature_C']),
                'capacity': np.ravel(cycle_struct['capacity_Ah']),
                'charge_stage': stages
            })
            
            all_cycles.append(temp_df)

        df_data = pd.concat(all_cycles, ignore_index=True)

        return df_data


    def _determine_charge_stages(self, current):
        
        threshold = 0.001
        stages = np.zeros_like(current, dtype=int)

        # 充/放电
        stages[current > threshold] = 1
        stages[current < -threshold] = 3

        # 静置: 以首次放电出现位置为界, 之前为充电后静置(2), 之后为放电后静置(4)
        rest_idx = np.where(np.abs(current) <= threshold)[0]
        if len(rest_idx) > 0:
            discharge_start_idx = np.where(stages == 3)[0]
            if len(discharge_start_idx) > 0:
                first_discharge = discharge_start_idx[0]
                stages[(np.abs(current) <= threshold) & (np.arange(len(current)) < first_discharge)] = 2
                stages[(np.abs(current) <= threshold) & (np.arange(len(current)) > first_discharge)] = 4
            else:
                stages[np.abs(current) <= threshold] = 2

        return stages

