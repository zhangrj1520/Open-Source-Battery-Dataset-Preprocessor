import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from data_processor.base_processor import BaseProcessor


class MitProcessor(BaseProcessor):
    FILTERED = {
        "2017-05-12": {8, 10, 12, 13, 22},           # 未达衰减到80%的cells
        "2017-06-30": {7, 8, 9, 15, 16},             # batch1延续的cells
        "2018-04-12": {2, 23, 32, 37, 42, 43},       # 噪声/问题cells
    }

    # batch1延续cell在batch2中的位置 batch1_idx: batch2_idx
    CONT_FILE = "2017-06-30_batchdata_updated_struct_errorcorrect.mat"
    CONTINUATION = {0: 7, 1: 8, 2: 9, 3: 15, 4: 16}
    
    
    def __init__(self, output_dir="data/processed", mat_path=None):
        super().__init__(output_dir)
        self.mat_path = mat_path
        self._f = None
        self._cont_f = None
        self._batch = None
        self._batch_date = None
        self._n_cells = 0


    def open(self, mat_path, cont_dir=None):
        """打开.mat文件, 返回有效电池索引列表"""
        self.mat_path = Path(mat_path)
        self._f = h5py.File(str(self.mat_path), 'r')
        self._batch = self._f['batch']
        self._n_cells = self._batch['barcode'].shape[0]

        # 从路径中提取batch日期
        fname = self.mat_path.name
        for date_key in self.FILTERED:
            if date_key in fname:
                self._batch_date = date_key
                break

        # 如果是batch1且提供了延续文件目录, 打开延续文件
        if self._batch_date == "2017-05-12" and cont_dir is not None:
            cont_path = Path(cont_dir) / self.CONT_FILE
            if cont_path.exists():
                self._cont_f = h5py.File(str(cont_path), 'r')

        skip = self.FILTERED.get(self._batch_date, set())
        valid = [i for i in range(self._n_cells) if i not in skip]
        return valid


    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None
        if self._cont_f is not None:
            self._cont_f.close()
            self._cont_f = None


    @property
    def n_cells(self):
        return self._n_cells


    def get_cell_name(self, index):
        return "#%d" % (index + 1)


    def load_data(self, cell_index):

        all_cycles = self._load_cycles(self._f, self._batch, cell_index, 1)

        # 对batch1的延续cell追加batch2的对应数据
        if (self._batch_date == "2017-05-12"
                and self._cont_f is not None
                and cell_index in self.CONTINUATION):
            cont_idx = self.CONTINUATION[cell_index]
            cont_batch = self._cont_f['batch']
            start_cycle = len(all_cycles) + 1
            extra = self._load_cycles(self._cont_f, cont_batch, cont_idx,
                                      start_cycle)
            if extra:
                all_cycles.extend(extra)

        df = pd.concat(all_cycles, ignore_index=True)
        df = self._remove_abnormal_cycles(df)
        df = self._compute_capacity(df)
        return df
    

    def _load_cycles(self, h5file, batch, cell_index, start_cycle):
        """ 加载单个 cell 的全部 cycles """
        cyc_ref = batch['cycles'][cell_index, 0]
        cyc_group = h5file[cyc_ref]
        n_cycles = cyc_group['I'].shape[0]

        cycles = []
        for ci in range(n_cycles):
            I = h5file[cyc_group['I'][ci, 0]][:].flatten()
            V = h5file[cyc_group['V'][ci, 0]][:].flatten()
            t = h5file[cyc_group['t'][ci, 0]][:].flatten() * 60
            T = h5file[cyc_group['T'][ci, 0]][:].flatten()
            Qc = h5file[cyc_group['Qc'][ci, 0]][:].flatten()
            Qd = h5file[cyc_group['Qd'][ci, 0]][:].flatten()

            stages = self._determine_charge_stages(I)

            temp_df = pd.DataFrame({
                'cycle_number': start_cycle + ci,
                'time': t - t[0],
                'voltage': V,
                'current': I,
                'temperature': T,
                'charge_capacity': Qc,
                'discharge_capacity': Qd,
                'charge_stage': stages,
            })
            cycles.append(temp_df)
        return cycles


    def _determine_charge_stages(self, I):
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
        """ 阶段完整 + 时间连续 (相邻数据点间隔 > 3600s 视为实验中断) """
        all_cycles = df['cycle_number'].unique()
        valid = pd.Series(True, index=all_cycles)

        stages_per_cycle = df.groupby('cycle_number')['charge_stage'].apply(
            lambda x: set(x.unique()))
        valid &= stages_per_cycle.apply(lambda s: {1, 3}.issubset(s))

        time_diff = df.groupby('cycle_number')['time'].diff()
        max_gap = time_diff.groupby(df['cycle_number']).max()
        valid &= max_gap < 3600

        valid_cycles = valid[valid].index.tolist()

        return df[df['cycle_number'].isin(valid_cycles)].copy()
