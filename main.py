import os
from pathlib import Path


if __name__ == "__main__":

    ## ---------------------------------------- XJTU Dataset ----------------------------------------
    # from data_processor.xjtu_processor import XjtuProcessor

    # current_threshold = 0.001
    # raw_data_dir = Path("E:/data/OpenSourceData/XJTU/")

    # for batch in ["Batch-1", "Batch-2", "Batch-3", "Batch-4", "Batch-5", "Batch-6"]:
   
    #     batch_dir = raw_data_dir / batch
    #     file_lst = os.listdir(batch_dir)

    #     processor = XjtuProcessor(output_dir=f"data/XJTU/{batch}/")

    #     for f in file_lst:
    #         file_path = batch_dir / f
    #         df = processor.load_data(file_path, current_threshold=current_threshold)
            
    #         cell_name = Path(f).stem
    #         processor.save_to_parquet(df, cell_name)
    
    ## ---------------------------------------- Tongji Dataset ----------------------------------------
    # from data_processor.tongji_processor import TongjiProcessor

    # raw_data_dir = Path("E:/data/OpenSourceData/Tongji/")

    # for dataset in ["Dataset_1_NCA_battery", "Dataset_2_NCM_battery", "Dataset_3_NCM_NCA_battery"]:

    #     dataset_dir = raw_data_dir / dataset
    #     file_lst = os.listdir(dataset_dir)

    #     processor = TongjiProcessor(output_dir=f"data/Tongji/{dataset}/")

    #     for f in file_lst:
    #         file_path = dataset_dir / f
    #         df = processor.load_data(file_path)

    #         cell_name = Path(f).stem
    #         processor.save_to_parquet(df, cell_name)

    ## ---------------------------------------- HUST Dataset ----------------------------------------
    # from data_processor.hust_processor import HustProcessor

    # raw_data_dir = Path("E:/data/OpenSourceData/HUST/")
    # file_lst = os.listdir(raw_data_dir)

    # processor = HustProcessor(output_dir="data/HUST/")

    # for f in file_lst:
    #     file_path = raw_data_dir / f
    #     df = processor.load_data(file_path)

    #     cell_name = Path(f).stem
    #     processor.save_to_parquet(df, cell_name)

    ## ---------------------------------------- NASA Dataset ----------------------------------------
    # from data_processor.nasa_processor import NasaProcessor

    # raw_data_dir = Path("E:/data/OpenSourceData/NASA/")

    # for group in [f"Group_{i}" for i in range(1, 7)]:
    #     group_dir = raw_data_dir / group
    #     file_lst = os.listdir(group_dir)
    #     file_lst = [f for f in file_lst if f.endswith('.mat')]

    #     processor = NasaProcessor(output_dir=f"data/NASA/{group}/")

    #     for f in file_lst:
    #         file_path = group_dir / f
    #         df = processor.load_data(file_path)

    #         cell_name = Path(f).stem
    #         processor.save_to_parquet(df, cell_name)

    ## ---------------------------------------- MIT Dataset ----------------------------------------
    from data_processor.mit_processor import MitProcessor

    mat_files = {
        "2017-05-12": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
        "2017-06-30": "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
        "2018-04-12": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
    }

    raw_data_dir = Path("E:/data/OpenSourceData/MIT/")

    for date_str, fname in mat_files.items():
        processor = MitProcessor(output_dir=f"data/MIT/{date_str}/")
        n_cells = processor.open(raw_data_dir / fname)
        print(f"  [{date_str}] {n_cells} batteries")

        for bi in range(n_cells):
            df = processor.load_data(bi)
            name = processor.get_cell_name(bi)
            processor.save_to_parquet(df, name)

        processor.close()