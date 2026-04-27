import scipy.io
import pandas as pd
import numpy as np

class BatteryProcessor:
    def __init__(self, file_path, threshold=0.001):
        self.file_path = file_path
        self.threshold = threshold
        self.raw_mat = scipy.io.loadmat(file_path)
        self.df_data = None
        self.df_summary = None

    def load_all_data(self):
        """
        一次性解析所有 data 结构并转换为 DataFrame
        """
        data_array = self.raw_mat['data'].squeeze()
        all_data_list = []
        
        print(f"正在展平 {len(data_array)} 个循环的数据...")
        for i, cycle_struct in enumerate(data_array):
            # 将每个 cycle 的数据转为 dict
            d = {
                'cycle_index': i + 1,
                'system_time': cycle_struct['system_time'].flatten(),
                'voltage': np.ravel(cycle_struct['voltage_V']),
                'current': np.ravel(cycle_struct['current_A']),
                'temperature': np.ravel(cycle_struct['temperature_C']),
                'capacity': np.ravel(cycle_struct['capacity_Ah']),
            }
            all_data_list.append(pd.DataFrame(d))
        
        # 核心优化：一次性 concat
        self.df_data = pd.concat(all_data_list, ignore_index=True)
        
        # 向量化时间处理：对整列进行转换，比在循环里快得多
        print("正在转换时间戳...")
        self.df_data['system_time'] = pd.to_datetime(self.df_data['system_time'], errors='coerce')
        
        # 计算相对时间（按 cycle 分组计算）
        self.df_data['relative_time_sec'] = self.df_data.groupby('cycle_index')['system_time'].transform(
            lambda x: (x - x.iloc[0]).dt.total_seconds()
        )
        return self.df_data

    def add_stage_column(self):
        """
        使用向量化逻辑加入 stage 列：
        1: 充电, 2: 充后静置, 3: 放电, 4: 放后静置
        """
        if self.df_data is None:
            raise ValueError("请先运行 load_all_data()")

        print("正在进行阶段判定...")
        df = self.df_data
        
        # 定义基础状态：1-充电，3-放电，0-静置
        df['charge_stage'] = 0
        df.loc[df['current'] > self.threshold, 'charge_stage'] = 1
        df.loc[df['current'] < -self.threshold, 'charge_stage'] = 3
        
        # 处理静置阶段 (2 和 4)
        # 逻辑：在一个 cycle 内，如果还没发生过放电，那它就是阶段 2；如果发生过放电，就是阶段 4
        def label_rest_phases(group):
            is_rest = group['charge_stage'] == 0
            # 找到放电开始的标志：当前及之前是否出现过阶段 3
            has_discharged = (group['charge_stage'] == 3).cummax()
            
            group.loc[is_rest & ~has_discharged, 'charge_stage'] = 2
            group.loc[is_rest & has_discharged, 'charge_stage'] = 4
            return group

        # 使用 groupby 处理每个 cycle 的静置逻辑
        self.df_data = df.groupby('cycle_index', group_keys=False).apply(label_rest_phases)
        return self.df_data

    def process_summary(self):
        """
        处理 summary 结构
        """
        summary_content = self.raw_mat['summary'][0, 0]
        summary_dict = {}
        for field in summary_content.dtype.names:
            val = np.ravel(summary_content[field])
            summary_dict[field] = val if len(val) > 1 else val[0]
        
        self.df_summary = pd.DataFrame(summary_dict)
        if 'cycle_index' not in self.df_summary.columns:
            self.df_summary.insert(0, 'cycle_index', range(1, len(self.df_summary) + 1))
        return self.df_summary

# --- 使用示例 ---
processor = BatteryProcessor('Batch1_Data.mat')
df_data = processor.load_all_data()
df_data = processor.add_stage_column()
df_summary = processor.process_summary()

print(df_data.head())
