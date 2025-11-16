原始数据集的目录结构
/Users/cuijinlong/Documents/datasets/pifubing
   |
    --optional_image    # 样本数据
         |  # 不同分类图片文件夹
         -- /cate1
         -- /cate2
         -- /cate3
         -- ...
         -- /caten
         |
         --data.xlsx  # 可选（可有可无）

分割数据集的目录结构(无xlsx)
/Users/cuijinlong/Documents/datasets/pifubing
   |
    --output_dataset_basic
         |  # 不同分类图片文件夹
         -- /test
             -- /cate1
             -- /cate2
             -- /cate3
             -- ...
             -- /caten
         -- /train
             -- /cate1
             -- /cate2
             -- /cate3
             -- ...
             -- /caten
         -- /val
             -- /cate1
             -- /cate2
             -- /cate3
             -- ...
             -- /caten
         --test_metadata.csv
         --train_metadata.csv
         --val_metadata.csv
         --dataset_info.txt
