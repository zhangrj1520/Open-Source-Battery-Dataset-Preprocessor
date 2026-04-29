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

    