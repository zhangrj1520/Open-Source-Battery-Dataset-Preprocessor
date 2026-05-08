import numpy as np
import pandas as pd
from data_processor.base_processor import BaseProcessor 


class TongjiProcessor(BaseProcessor):
    """处理Tongji数据集的Processor"""
    def load_data(self, file_path):
        """加载 Tongji .csv 数据并执行标准化转换"""
        raw_df = pd.read_csv(file_path)
        
        # 提取控制信号
        control_mA = raw_df['control/mA'].fillna(0).values
        control_V = raw_df['control/V'].fillna(0).values
        
        # 判定充放电状态
        stages = self._determine_charge_stages(control_mA, control_V)

        # 将充放电容量列合并
        q_charge = raw_df['Q charge/mA.h'].fillna(0).values
        q_discharge = raw_df['Q discharge/mA.h'].fillna(0).values
        merged_capacity_Ah = (q_charge + q_discharge) / 1000.0

        # 创建DataFrame
        df_data = pd.DataFrame({
            'cycle_number': raw_df['cycle number'].values,       
            'time': raw_df['time/s'].values,                    
            'voltage': raw_df['Ecell/V'].values,
            'current': raw_df['<I>/mA'].values / 1000.0,        
            'capacity': merged_capacity_Ah,
            'charge_stage': stages
        })

        # 每个cycle的时间从0开始
        cycle_start_times = df_data.groupby('cycle_number')['time'].transform('first')
        df_data['time'] = df_data['time'] - cycle_start_times

        # 异常cycle剔除
        df_data = self._remove_abnormal_cycles(file_path, df_data)
     
        return df_data
    

    def _determine_charge_stages(self, control_mA, control_V):
        """
        Tongji dataset 的阶段判定逻辑：
        利用控制信号 control/mA 和 control/V 进行精确划分。
        - 1: 充电 (control_mA > 0 或 control_V > 0)
        - 2: 充电后静置
        - 3: 放电 (control_mA < 0)
        - 4: 放电后静置
        """
        s_stages = pd.Series(np.zeros(len(control_mA), dtype=int))
        
        # 判定充放电状态
        is_charge = (control_mA > 1e-5) | (control_V > 1e-5)
        is_discharge = (control_mA < -1e-5)
        is_rest = ~(is_charge | is_discharge)
        
        s_stages[is_charge] = 1
        s_stages[is_discharge] = 3
        s_stages[is_rest] = np.nan  
        
        # 判定静置状态
        last_active_stage = s_stages.ffill().fillna(1)
        rest_stages = np.where(last_active_stage == 1, 2, 4)
        
        # 填回真实静置状态
        s_stages[is_rest] = rest_stages[is_rest]
        
        return s_stages.astype(int).values
    

    def _remove_abnormal_cycles(self, file_path, df):
        """
        多维异常cycle检测:
        1. 放电容量过滤 (Dataset-specific)
        2. 阶段完整性: 每个cycle应有恰好4段(charge → rest → discharge → rest)
        3. 时间连续性: 相邻数据点间隔 < 600s
        """
        df_discharge = df[df['charge_stage'] == 3]
        cycle_caps = df_discharge.groupby('cycle_number')['capacity'].max()
        all_cycles = cycle_caps.index

        # 放电容量过滤
        capacity_valid = pd.Series(False, index=all_cycles)

        if "Dataset_1_NCA_battery" in str(file_path):
            capacity_valid = (cycle_caps >= 2.5) & (cycle_caps <= 3.5)

        elif "Dataset_2_NCM_battery" in str(file_path):
            capacity_valid = cycle_caps >= 2.5

        elif "Dataset_3_NCM_NCA_battery" in str(file_path):
            valid_list = []
            q_p = cycle_caps.iloc[0]
            delta = 1
            for cyc, q_dis_ah in cycle_caps.items():
                if q_dis_ah < 1.65 or q_dis_ah > 2.51:
                    delta += 1
                    continue
                if abs(q_dis_ah - q_p) > delta * 0.01:
                    delta += 1
                    continue
                valid_list.append(cyc)
                q_p = q_dis_ah
                delta = 1
            capacity_valid = pd.Series(all_cycles.isin(valid_list), index=all_cycles)
        else:
            capacity_valid[:] = True

        # 阶段完整性: 每个cycle应有恰好4次阶段切换
        stage_shifted = df.groupby('cycle_number')['charge_stage'].shift()
        stage_changed = (df['charge_stage'] != stage_shifted)
        transitions_per_cycle = stage_changed.groupby(df['cycle_number']).sum()
        stage_valid = transitions_per_cycle == 4

        # 时间连续性: 相邻数据点间隔 < 600s
        time_diff = df.groupby('cycle_number')['time'].diff()
        max_gap_per_cycle = time_diff.groupby(df['cycle_number']).max()
        time_valid = max_gap_per_cycle < 600

        # 过滤条件取交集
        final_valid = capacity_valid & stage_valid & time_valid
        valid_cycle_numbers = final_valid[final_valid].index.tolist()

        return df[df['cycle_number'].isin(valid_cycle_numbers)].copy()

    