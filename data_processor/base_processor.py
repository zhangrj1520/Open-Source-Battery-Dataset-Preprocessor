import pathlib
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class BaseProcessor(ABC):
    """所有电池数据集解析器的抽象基类"""
    def __init__(self, output_dir="data/processed"):
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


    @abstractmethod
    def load_data(self, file_path, **kwargs):
        """
        原始数据加载与合并。
        返回标准化的 DataFrame, 包含统一列名: 'cycle_number', 'time', 'voltage', 'current', 'capacity', 'charge_state'等
        具体逻辑由特定数据集的Processor根据数据特点自行实现
        """
        pass


    @abstractmethod
    def _determine_charge_stages(self, *args, **kwargs):
        """
        判定充放电阶段: 1, 2, 3, 4 分别代表 充电、充电后静置、放电、放电后静置
        具体逻辑由特定数据集的Processor根据数据特点自行实现
        """
        pass


    def save_to_parquet(self, df, filename):
        """将标准化后的 DataFrame 导出为 Parquet 格式"""
        output_path = self.output_dir / f"{filename}.parquet"
        df.to_parquet(output_path, engine='pyarrow', index=False)
        print(f"✅ Successfully saved at {output_path}")
