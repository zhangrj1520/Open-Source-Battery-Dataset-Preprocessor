import os
from pathlib import Path


if __name__ == "__main__":

    ## ---------------------------------------- XJTU Dataset ----------------------------------------
    from data_processor.xjtu_processor import XjtuProcessor

    current_threshold = 0.001
    raw_data_dir = Path("E:/data/OpenSourceData/XJTU/")

    for batch in ["Batch-1", "Batch-2", "Batch-3", "Batch-4", "Batch-5", "Batch-6"]:
   
        batch_dir = raw_data_dir / batch
        file_lst = os.listdir(batch_dir)

        processor = XjtuProcessor(output_dir=f"data/XJTU/{batch}/")

        for f in file_lst:
            file_path = batch_dir / f
            df = processor.load_data(file_path, current_threshold=current_threshold)
            
            cell_name = Path(f).stem
            processor.save_to_parquet(df, cell_name)
    
    ## ---------------------------------------- Tongji Dataset ----------------------------------------
    from data_processor.tongji_processor import TongjiProcessor

    raw_data_dir = Path("E:/data/OpenSourceData/Tongji/")

    for dataset in ["Dataset_1_NCA_battery", "Dataset_2_NCM_battery", "Dataset_3_NCM_NCA_battery"]:

        dataset_dir = raw_data_dir / dataset
        file_lst = os.listdir(dataset_dir)

        processor = TongjiProcessor(output_dir=f"data/Tongji/{dataset}/")

        for f in file_lst:
            file_path = dataset_dir / f
            df = processor.load_data(file_path)
            
            cell_name = Path(f).stem
            processor.save_to_parquet(df, cell_name)