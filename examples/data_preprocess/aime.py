# import argparse
# import os

# import datasets

# from verl.utils.hdfs_io import copy, makedirs



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--local_dir", default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/yuechuhuai/rldata/RL_evaldata/HuggingFaceH4/aime_2024")
#     parser.add_argument("--hdfs_dir", default=None)

#     args = parser.parse_args()

#     # 使用指定路径加载数据集
#     data_source = "aime24"  
#     dataset_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/yuechuhuai/rldata/RL_evaldata/HuggingFaceH4/aime_2024/data"
    
#     # 加载jsonl文件数据集
#     dataset = datasets.load_dataset(dataset_path, trust_remote_code=True)

#     test_dataset = dataset["test"]

#     instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."

#     # 添加唯一标识的行
#     def make_map_fn():
#         def process_fn(example, idx):
#             question = example.pop("problem")

#             # 在问题后附加说明
#             question = question + " " + instruction_following

#             answer = example.pop("answer")


#             # 创建最终数据结构
#             data = {
#                 "data_source": data_source,
#                 "prompt": [{"role": "user", "content": question}],
#                 "ability": "math",
#                 "reward_model": {"style": "rule", "ground_truth": answer},
#                 "extra_info": {"split": "test", "index": idx},
#             }
#             return data

#         return process_fn

#     # 处理数据集并映射转换
#     test_dataset = test_dataset.map(function=make_map_fn(), with_indices=True)

#     # 本地保存目录
#     local_dir = args.local_dir
#     hdfs_dir = args.hdfs_dir

#     # 转换为Parquet格式并保存
#     test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

#     if hdfs_dir is not None:
#         # 如果提供了HDFS路径，创建目录
#         makedirs(hdfs_dir)

#         # 将文件复制到HDFS
#         copy(src=local_dir, dst=hdfs_dir)


import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/yuechuhuai/rldata/RL_evaldata/yentinglin/aime_2025")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # 使用指定路径加载数据集
    data_source = "aime25"  
    dataset_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/yuechuhuai/rldata/RL_evaldata/yentinglin/aime_2025/data"
    
    # 加载jsonl文件数据集
    dataset = datasets.load_dataset(dataset_path, trust_remote_code=True)

    test_dataset = dataset["test"]

    instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."

    # 添加唯一标识的行
    def make_map_fn():
        def process_fn(example, idx):
            question = example.pop("problem")

            # 在问题后附加说明
            question = question + " " + instruction_following

            answer = example.pop("answer")


            # 创建最终数据结构
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {"split": "test", "index": idx},
            }
            return data

        return process_fn

    # 处理数据集并映射转换
    test_dataset = test_dataset.map(function=make_map_fn(), with_indices=True)

    # 本地保存目录
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # 转换为Parquet格式并保存
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        # 如果提供了HDFS路径，创建目录
        makedirs(hdfs_dir)

        # 将文件复制到HDFS
        copy(src=local_dir, dst=hdfs_dir)
