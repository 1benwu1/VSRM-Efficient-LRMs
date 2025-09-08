import torch
import itertools
import numpy as np
from verl import DataProto
import torch.distributed as dist
from tensordict import TensorDict
from dprint import dprint,wprint
import re
import time
from vllm import LLM, SamplingParams
from collections import defaultdict
from verl.utils.reward_score.math import compute_score



class roll_postprocess:
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer  
        self.config = config.rollout
        self.num_ans = self.config['n2']
        self.top_p = self.config.top_p
        self.temperature = self.config.temperature
        self.total_res_len = self.config.response_length
        self.total_prom_len = self.config.prompt_length

        dprint("roll_postprocess initializing")
        
        self.high_entropy_words = [
            "wait", "Wait", "WAIT",
            "but", "But", "BUT",
            "alternatively", "Alternatively", "ALTERNATIVELY",
            "however", "However", "HOWEVER",
            "hmm", "Hmm", "HMM",
            "Let me double-check", "let me double-check",
            "But wait", "but wait", 
            "Hold on", "hold on", 
            "Looking back", "looking back", 
            "Seems solid", "seems solid", 
            "Similarly", "similarly",
            "That's correct, but", "that's correct, but", 
            "That seems right", "that seems right",
            "Wait, but", "wait, but",
            "So"
        ]
        
        # 预编译正则表达式用于高效匹配
        sorted_words = sorted(self.high_entropy_words, key=len, reverse=True)
        pattern = '|'.join(map(re.escape, sorted_words))
        self.high_entropy_regex = re.compile(pattern)
        
        # 句子结束符正则（用于精确切分）
        self.sentence_end_regex = re.compile(r'[.!?\n]')
        self.think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)  

        self.min_think_chars = 1000  
        self.min_split_interval = 400  

        self.unsplit_log_file = "unsplit_trajectories.log"
        dprint(f"无法切分的主轨迹将记录到: {self.unsplit_log_file}")

    def postprocess_output(self, generated_seq: DataProto, rollouter, GTs):
        dprint("postprocess_output starts!!!!!!")
        
        self.generated_seq = generated_seq

        # 处理GTs
        re_GTs = [item['ground_truth'] for item in GTs for _ in range(self.config['n'])]
        
        rollouts_tensors = generated_seq.batch["input_ids"]

        prompts_tensors = generated_seq.batch["prompts"]

        responses_tensors = generated_seq.batch["responses"]

        attention_mask = generated_seq.batch["attention_mask"]


        all_subrolls = []
        unsplit_indices = [] 
        unsplit_texts = []    

        token_positions = [] 

        response_valid_lengths = [] 
        nopd_prompt_texts = []         
        pd_prompt_texts = [] 


        for rollout_idx in range(len(responses_tensors)):

            prompt_mask = attention_mask[rollout_idx, :prompts_tensors.shape[1]]
            prompt_valid_length = prompt_mask.sum().item()
            prompt_ids = prompts_tensors[rollout_idx, -prompt_valid_length:] if prompt_valid_length > 0 else []
            nopd_prompt_texts.append(self.tokenizer.decode(prompt_ids, skip_special_tokens=True))
            
            response_mask = attention_mask[rollout_idx, prompts_tensors.shape[1]:]
            response_valid_lengths.append(response_mask.sum().item())

            response_tensor = responses_tensors[rollout_idx]
            response_valid_length = response_valid_lengths[rollout_idx]
            nopd_prompt_text = nopd_prompt_texts[rollout_idx]

            valid_response = response_tensor[:response_valid_length]
            response_text = self.tokenizer.decode(valid_response, skip_special_tokens=True)
            
            positions = self.find_high_entropy(response_text)
            
            token_mapping = self.get_token_positions(response_text)
            
            subrolls = []
            subroll_token_positions = [] 
            
            for position in positions:

                # 创建子轨迹
                subroll = response_text[:position] + "</think>"

                subrolls.append(nopd_prompt_text + subroll)
                
                # 找到对应的token位置
                token_idx = self.find_token_index_for_char(position, token_mapping)

                if token_idx is not None:
                    subroll_token_positions.append(token_idx)
                else:
                    dprint(f"警告: 无法为字符位置 {position} 找到对应的token位置")


            subrolls.append(nopd_prompt_text + response_text[:response_text.find("</think>") + len("</think>")]) 


            all_subrolls.append(subrolls)
            token_positions.append(subroll_token_positions)  # 存储当前主轨迹的所有切分点
            
            if len(subrolls) == 0:
                # dprint(nopd_prompt_text + response_text)
                unsplit_indices.append(rollout_idx)
                unsplit_texts.append(nopd_prompt_text + response_text)

        # dprint("主轨迹平均长度：")
        # dprint(sum(response_valid_lengths)/len(response_valid_lengths))
        # # 打印统计信息
        # dprint(f"Total rollouts processed: {len(all_subrolls)}")
        # dprint(f"Subrollouts per rollout: {[len(subs) for subs in all_subrolls]}")
        # dprint(f"SUM of Subrollouts: {sum([len(subs) for subs in all_subrolls])}")
        
        if unsplit_texts:
            self.log_unsplit_trajectories(unsplit_indices, unsplit_texts)
        
        responses = self.batch_generate(all_subrolls, rollouter)
        subroll_acc = self.calculate_subtrajectory_scores(responses, re_GTs)

        step_reward_tensor=self.compute_step_reward(subroll_acc, token_positions,response_valid_lengths)

        generated_seq.batch["step_reward_tensor"] = step_reward_tensor

        return generated_seq









    def get_token_positions(self, text):

        encoding = self.tokenizer(text, return_offsets_mapping=True)
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        return list(zip(tokens, encoding["offset_mapping"]))

    def find_token_index_for_char(self, char_pos, token_mapping):

        for idx, (token, (start, end)) in enumerate(token_mapping):
            if start <= char_pos < end:
                return idx 
        return None
    

    def find_high_entropy(self, rollout):
        start_time = time.time()
        
        match = self.think_pattern.search(rollout)
        
        if not match:
            incomplete_pattern = r'<think>(.*)'
            match = re.search(incomplete_pattern, rollout, re.DOTALL)
        
        if not match:
            elapsed_ms = (time.time() - start_time) * 1000
            dprint(f"find_high_entropy executed in {elapsed_ms:.2f} ms")
            return []
        
        think_start = match.start(1)  
        think_end = match.end(1)      
        
        positions = set()
        last_valid_pos = -1 
        
        for match in self.high_entropy_regex.finditer(rollout, think_start, think_end):
            word_start = match.start()
            word_end = match.end()
            
            if word_start - think_start < self.min_think_chars:
                continue
                
            if last_valid_pos != -1 and (word_start - last_valid_pos) < self.min_split_interval:
                continue
                
            sentence_end_pos = -1
            for pos in range(word_start - 1, think_start - 1, -1):
                if self.sentence_end_regex.match(rollout[pos]):
                    sentence_end_pos = pos + 1  
                    break
            
            if sentence_end_pos > 0:
                positions.add(sentence_end_pos)
                last_valid_pos = sentence_end_pos
            else:
                positions.add(word_start)
                last_valid_pos = word_start
        
        elapsed_ms = (time.time() - start_time) * 1000
        # dprint(f"find_high_entropy executed in {elapsed_ms:.2f} ms")
        
        return sorted(positions)

    
    def create_sampling_params(self):
        return SamplingParams(
            n=self.num_ans,
            max_tokens=768,  
            temperature=self.temperature,
            top_p=self.top_p,
        )
    

    def batch_generate(self, trajectories, rollout):
        start_time = time.time()
        
        results = [defaultdict(list) for _ in range(len(trajectories))]
        
        vllm_inputs = []
        index_mapping = []  
        
        # 遍历每个主轨迹
        for main_idx, subrolls in enumerate(trajectories):
            # 如果这个主轨迹没有子轨迹，跳过生成
            if not subrolls:
                dprint(f"主轨迹 {main_idx} 没有子轨迹，跳过生成")
                continue
            
            for sub_idx, sub_traj in enumerate(subrolls):
                input_ids = self.tokenizer(
                    sub_traj, 
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length = self.total_res_len + self.total_prom_len
                ).input_ids[0].tolist()
                
                vllm_inputs.append({
                    "prompt_token_ids": input_ids,
                    "meta": (main_idx, sub_idx)
                })
                index_mapping.append((main_idx, sub_idx))
        
        if not vllm_inputs:
            elapsed_s = time.time() - start_time
            dprint(f"subrollout cost {elapsed_s:.2f} s (没有子轨迹需要生成)")
            return results
        
        sampling_params = self.create_sampling_params()
        
        outputs = rollout.inference_engine.generate(
            prompts=vllm_inputs,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        
        for output, (main_idx, sub_idx) in zip(outputs, index_mapping):
            for sample in output.outputs:
                decoded_text = self.tokenizer.decode(sample.token_ids)
                results[main_idx][sub_idx].append({
                    "tokens": sample.token_ids,
                    "text": decoded_text.strip()
                })

        elapsed_s = time.time() - start_time
        dprint(f"subrollout cost {elapsed_s:.2f} s")

        return results
    

    def calculate_subtrajectory_scores(self, responses, gts):
        if len(gts) != len(responses):
            dprint(f"错误: GTs数量({len(gts)})与主轨迹数量({len(responses)})不匹配")
            return []
        
        if not responses:
            return []
        
        main_trajectory_scores = []
        
        for main_idx, main_responses in enumerate(responses):
            current_gt = gts[main_idx]
            sub_scores = []  
            
            # # 如果没有子轨迹，添加0.0作为占位符
            # if not main_responses:
            #     dprint(f"主轨迹 {main_idx} 没有子轨迹")
            #     sub_scores.append(None)
            #     main_trajectory_scores.append(sub_scores)
            #     continue
            
            sorted_sub_indices = sorted(main_responses.keys())
            
            for sub_idx in sorted_sub_indices:
                response_list = main_responses[sub_idx]
                scores = []
                
                for response in response_list:
                    score = compute_score(response["text"], current_gt)
                    scores.append(score)
                
                avg_score = sum(scores) / len(scores) if scores else 0.0
                sub_scores.append(avg_score)
            
            main_trajectory_scores.append(sub_scores)
        
        return main_trajectory_scores




    def compute_step_reward(self, subroll_acc, token_positions, response_valid_lengths):

        reward_tensor_length = len(self.generated_seq.batch["responses"][0])

        gamma = 0.7  
        MAX_LOOKAHEAD = 4  
        
        batch_size = len(token_positions)

        max_response_length = reward_tensor_length
        
        step_reward_tensors = torch.zeros((batch_size, max_response_length), dtype=torch.float32)  

        for i in range(batch_size):
            valid_length = response_valid_lengths[i]
            if valid_length == 0:
                continue
                
            pos_list = token_positions[i]
            acc_list = subroll_acc[i]
            
            if not pos_list:
                continue
                
            if len(acc_list) != len(pos_list) + 1:
                print(f"数据不匹配：样本{i} - 切分点{len(pos_list)}个，正确率{len(acc_list)}个")
                continue
                
            for j, token_idx in enumerate(pos_list):
                current_acc = acc_list[j]
                next_acc = acc_list[j+1]
                
                delta = next_acc - current_acc
                
                if abs(delta) < 1e-5: 
                    k = j + 1
                    steps = 0
                    
                    while (steps < MAX_LOOKAHEAD and 
                        k < len(pos_list) and 
                        abs(acc_list[k+1] - acc_list[k]) < 1e-5):
                        k += 1
                        steps += 1
                    
                    if steps == MAX_LOOKAHEAD:  
                        trend_reward = 0.0
                    elif k < len(pos_list):  
                        change_delta = acc_list[k+1] - acc_list[k]
                        sign = 1 if change_delta > 0 else -1
                        distance = k - j
                        trend_reward = sign * abs(change_delta) * (gamma ** distance)
                    else:  # 到达序列末尾
                        final_delta = acc_list[-1] - current_acc
                        sign = 1 if final_delta > 0 else -1
                        distance = len(pos_list) - j - 1
                        trend_reward = sign * abs(final_delta) * (gamma ** distance)
                        
                    reward = trend_reward
                else:
                    reward = delta
                    
                if 0 <= token_idx < valid_length:
                    step_reward_tensors[i, token_idx] = reward
                else:
                    print(f"警告：样本{i}位置{token_idx}超出有效长度{valid_length}")

        return step_reward_tensors



    def log_unsplit_trajectories(self, indices, texts):

        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"unsplit_trajectories.log"
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"===== 无法切分的主轨迹报告 =====\n")
                f.write(f"生成时间: {timestamp}\n")
                f.write(f"总数: {len(indices)}\n\n")
                
                for idx, text in zip(indices, texts):
                    f.write(f"=== 主轨迹 {idx} ===\n")
                    f.write(f"长度: {len(text)} 字符\n")
                    f.write(f"内容:\n{text}\n")
                    f.write("\n" + "="*50 + "\n\n")
            
            dprint(f"已记录 {len(indices)} 个无法切分的主轨迹到 {filename}")
        except Exception as e:
            dprint(f"记录无法切分轨迹时出错: {str(e)}")

