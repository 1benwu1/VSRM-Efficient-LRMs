import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/mnt/dolphinfs/hdd_pool/docker/share/yuechuhuai/rldata/RL_trainingdata/DeepScaleR-Preview-Dataset")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    # 使用指定路径加载数据集
    data_source = "Deepscalar"  
    dataset_path = "/mnt/dolphinfs/hdd_pool/docker/share/yuechuhuai/rldata/RL_trainingdata/DeepScaleR-Preview-Dataset/deepscaler.json"
    
    # 加载jsonl文件数据集
    dataset = datasets.load_dataset('json', data_files={'train': dataset_path})

    train_dataset = dataset["train"]

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
                "extra_info": {"split": "train", "index": idx},
            }
            return data

        return process_fn

    # 处理数据集并映射转换
    train_dataset = train_dataset.map(function=make_map_fn(), with_indices=True)

    # 本地保存目录
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # 转换为Parquet格式并保存
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))

    if hdfs_dir is not None:
        # 如果提供了HDFS路径，创建目录
        makedirs(hdfs_dir)

        # 将文件复制到HDFS
        copy(src=local_dir, dst=hdfs_dir)
