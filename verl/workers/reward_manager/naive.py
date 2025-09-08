# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from dprint import dprint

class NaiveRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score 
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False, state = "Train",proj_n=None, exp_n=None):
        """We will expand this function gradually based on the available datasets"""
        self.proj_n=proj_n
        self.exp_n=exp_n

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if state=="val":

            if "rm_scores" in data.batch.keys():
                if return_dict:
                    return {"reward_tensor": data.batch["rm_scores"]}
                else:
                    return data.batch["rm_scores"]

            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            reward_extra_info = defaultdict(list)

            already_print_data_sources = {}


            batch_categories = defaultdict(int)
            total_score = 0.0


            batch_res_lenghth=[]

            grouped_stats = defaultdict(lambda: {
                'scores': [],
                'categories': defaultdict(int),
                'lengths': [],
                'count': 0
            })

            
            for i in range(len(data)): 

                data_item = data[i]  # DataProtoItem

                prompt_ids = data_item.batch["prompts"]

                prompt_length = prompt_ids.shape[-1]

                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                batch_res_lenghth.append(valid_response_length)
                # decode
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

                data_source = data_item.non_tensor_batch[self.reward_fn_key]




                extra_info = data_item.non_tensor_batch.get("extra_info", None)
                # print("---------prompt_str--------") 
                # print(prompt_str)
                # print("---------response_str--------")
                # print(response_str)
                # print("---------ground_truth--------")
                # print(ground_truth)            
                # print("-----------------------------")
                score,category = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    solution_len=len(valid_response_ids),
                    extra_info=extra_info,
                )

                if isinstance(score, dict):
                    reward = score["score"]
                    # Store the information including original reward
                    for key, value in score.items():
                        reward_extra_info[key].append(value)
                else:
                    reward = score

                batch_categories[category] += 1
                total_score += score

                reward_tensor[i, valid_response_length - 1] = reward  

                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    # print("[prompt]", prompt_str)
                    # print("[response]", response_str)
                    # print("[ground_truth]", ground_truth)
                    if isinstance(score, dict):
                        for key, value in score.items():
                            print(f"[{key}]", value)
                    else:
                        print("[score]", score)


                group_key = data_source
                grouped_stats[group_key]['scores'].append(reward)
                grouped_stats[group_key]['categories'][category] += 1
                grouped_stats[group_key]['lengths'].append(valid_response_length)  
                grouped_stats[group_key]['count'] += 1

            avg_length = torch.stack(batch_res_lenghth).float().mean()  
            print(f"平均长度: {avg_length.item():.2f}")

            filename = f"{self.proj_n}_{self.exp_n}.txt"
            filename = "path"+filename
            report_lines = []  

            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            report_lines.append("\n" + "="*50)
            report_lines.append(f" 批量验证报告 @ {timestamp}")
            report_lines.append("="*50)
            
            # 遍历所有数据源
            for data_source_key, stats in grouped_stats.items():
                total_score = sum(stats['scores'])
                batch_size = stats['count']
                avg_length = torch.stack(stats['lengths']).float().mean() if batch_size > 0 else 0

                source_report = self._generate_single_report(
                    data_source=data_source_key,
                    batch_size=batch_size,
                    total_score=total_score,
                    categories=stats['categories'],
                    avg_length=avg_length
                )
                report_lines.extend(source_report)
                report_lines.append("")  # 添加空行分隔
            
            for line in report_lines:
                print(line)

            try:
                with open(filename, "a", encoding="utf-8") as f:
                    f.write("\n".join(report_lines))
                    f.write("\n\n")  
                print(f"\n报告已保存至: {filename}")
            except Exception as e:
                print(f"\n保存报告失败: {str(e)}")


            if return_dict:
                return {
                    "reward_tensor": reward_tensor,
                    "reward_extra_info": reward_extra_info,
                }
            else:
                return reward_tensor







        else: # train

            if "rm_scores" in data.batch.keys():
                if return_dict:
                    return {"reward_tensor": data.batch["rm_scores"]}
                else:
                    return data.batch["rm_scores"]

            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            reward_extra_info = defaultdict(list)

            already_print_data_sources = {}

            for i in range(len(data)): 
                data_item = data[i]  # DataProtoItem

                prompt_ids = data_item.batch["prompts"]

                prompt_length = prompt_ids.shape[-1]

                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                # decode
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

                data_source = data_item.non_tensor_batch[self.reward_fn_key]

                extra_info = data_item.non_tensor_batch.get("extra_info", None)

                score = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    solution_len=len(valid_response_ids),
                    extra_info=extra_info,
                )
                print("训练时reward已经算完")
                if isinstance(score, dict):
                    reward = score["score"]
                    # Store the information including original reward
                    for key, value in score.items():
                        reward_extra_info[key].append(value)
                else:
                    reward = score


                # # 把之前算好的step reward和基于结果的reward拼起来
                step_wise_reward = data_item.batch["step_reward_tensor"]


                reward_tensor[i] = step_wise_reward
                reward_tensor[i, valid_response_length - 1] = reward 

                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    # print("[prompt]", prompt_str)
                    # print("[response]", response_str)
                    # print("[ground_truth]", ground_truth)
                    if isinstance(score, dict):
                        for key, value in score.items():
                            print(f"[{key}]", value)
                    else:
                        print("[score]", score)

            if return_dict:
                return {
                    "reward_tensor": reward_tensor,
                    "reward_extra_info": reward_extra_info,
                }
            else:
                return reward_tensor




    def _generate_single_report(self, data_source, batch_size, total_score, categories, avg_length):
        """生成单个数据源的报告内容（不执行打印）"""
        avg_score = total_score / batch_size if batch_size > 0 else 0
        
        # 分类映射表
        CATEGORY_MAP = {
            "complete_think_no_boxed": "有完整think,但没有boxed",
            "complete_think_boxed_ans_parse_fail": "有完整think,boxed内的答案解析有误",
            "complete_think_ans_correct": "有完整think,答案正确",
            "complete_think_ans_wrong": "有完整think,答案错误",
            "incomplete_think_no_boxed": "没有完整think,也没有boxed",
            "incomplete_think_boxed_ans_parse_fail": "没有完整think,boxed内的答案解析有误",
            "incomplete_think_ans_correct": "不完整think,答案正确",
            "incomplete_think_ans_wrong": "不完整think,答案错误",
            "no_think_no_boxed": "没think,也没有boxed",
            "no_think_boxed_ans_parse_fail": "没有think,boxed内的答案解析有误",
            "no_think_ans_correct": "无think,答案正确",
            "no_think_ans_wrong": "无think,答案错误",
            "error": "处理异常",
            "unknown": "未知分类",
        }
        
        # 构建报告内容
        lines = []
        lines.append(f"[数据源: {data_source}]")
        lines.append(f"- 样本数量: {batch_size}")
        lines.append(f"- 平均得分: {avg_score:.4f}")
        lines.append(f"- 平均长度: {avg_length:.2f}")
        lines.append("")
        
        # 分类统计
        for cat_id in range(1, 13):
            cat_name = [
                "complete_think_no_boxed", 
                "complete_think_boxed_ans_parse_fail", 
                "complete_think_ans_correct",
                "complete_think_ans_wrong",
                "incomplete_think_no_boxed", 
                "incomplete_think_boxed_ans_parse_fail", 
                "incomplete_think_ans_correct",
                "incomplete_think_ans_wrong",
                "no_think_no_boxed", 
                "no_think_boxed_ans_parse_fail", 
                "no_think_ans_correct",
                "no_think_ans_wrong",
            ][cat_id-1]
            
            count = categories.get(cat_name, 0)
            percentage = (count / batch_size) * 100 if batch_size > 0 else 0
            lines.append(f"{cat_id:02d}. {CATEGORY_MAP[cat_name]}: {count} ({percentage:.1f}%)")
        
        # 添加其他类别
        other_cats = [("error", "处理异常"), ("unknown", "未知分类")]
        for cat_key, cat_name in other_cats:
            count = categories.get(cat_key, 0)
            if count > 0:
                percentage = (count / batch_size) * 100 if batch_size > 0 else 0
                lines.append(f" - {cat_name}: {count} ({percentage:.1f}%)")
        
        lines.append("-" * 50)
        return lines