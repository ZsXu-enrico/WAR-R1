import os
import json
from pathlib import Path
import torch
from typing import Dict, List
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, TaskType
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Trainer\\.tokenizer is now deprecated.*")
import random
import re
import gc
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
print("CUDA available:", torch.cuda.is_available())
print("Number of CUDA devices:", torch.cuda.device_count())
if torch.cuda.device_count() > 0:
    print("Device name:", torch.cuda.get_device_name(0))
else:
    print("No CUDA devices detected")

class APIRecommendationGRPOTrainer:
    """
    API recommendation trainer using TRL's GRPOTrainer with LoRA implementation
    Modified to match the first file's data handling approach
    """
    def __init__(
        self,
        model_path: str,
        api_repo_path: str = 'used_api_list.json',
        device: str = None,
        use_lora: bool = True,
        lora_r: int = 32,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: List[str] = None
    ):
        # Device setup - using cuda:0 to match the first file
        self.DEVICE = torch.device("cuda:0")
        print(f"Using device: {self.DEVICE}")

        # LoRA configuration
        self.use_lora = use_lora
        self.lora_config = None
        if use_lora:
            # Default target modules for common transformer architectures
            if target_modules is None:
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            
            self.lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                # use_dora=True,
                bias="none"
            )
            print(f"LoRA enabled with r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
            print(f"Target modules: {target_modules}")

        # Tokenizer setup
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding=False)
        
        # Special tokens - matching the first file exactly
        self.EOS_TOKEN = self.tokenizer.eos_token
        self.STOP_NAME = "STOP"
        self.API_STOP_TOKEN = f"<API_{self.STOP_NAME}>"
        
        # Add API_START token
        self.DES_START_TOKEN = "<DES_START>"
        self.DES_STOP_TOKEN = "<DES_STOP>"
        self.API_START_TOKEN = "<API_START>"
        self.REASON_START_TOKEN = "<REASON_START>"
        self.REASON_STOP_TOKEN = "<REASON_STOP>"
        
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": [
                self.API_START_TOKEN, self.API_STOP_TOKEN,
                self.DES_START_TOKEN, self.DES_STOP_TOKEN,
                self.REASON_START_TOKEN, self.REASON_STOP_TOKEN,
            ]
        })

        # Get token IDs
        self.API_START_ID = self.tokenizer.convert_tokens_to_ids(self.API_START_TOKEN)
        self.API_STOP_ID = self.tokenizer.convert_tokens_to_ids(self.API_STOP_TOKEN)
        self.DES_START_ID = self.tokenizer.convert_tokens_to_ids(self.DES_START_TOKEN)
        self.DES_STOP_ID = self.tokenizer.convert_tokens_to_ids(self.DES_STOP_TOKEN)
        self.REASON_START_ID = self.tokenizer.convert_tokens_to_ids(self.REASON_START_TOKEN)
        self.REASON_STOP_ID = self.tokenizer.convert_tokens_to_ids(self.REASON_STOP_TOKEN)
        self.EOS_TOKEN_ID=self.tokenizer.convert_tokens_to_ids(self.EOS_TOKEN)
        print(f"Added API_START token: {self.API_START_TOKEN}, ID: {self.API_START_ID}")
        print(f"Added API_STOP token: {self.API_STOP_TOKEN}, ID: {self.API_STOP_ID}")
        print(f"Added DES_START token: {self.DES_START_TOKEN}, ID: {self.DES_START_ID}")
        print(f"Added DES_STOP token: {self.DES_STOP_TOKEN}, ID: {self.DES_STOP_ID}")
        print(f"Added REASON_START token: {self.REASON_START_TOKEN}, ID: {self.REASON_START_ID}")
        print(f"Added REASON_STOP token: {self.REASON_STOP_TOKEN}, ID: {self.REASON_STOP_ID}")
        

        self.prompt_template = (
            "### API Recommendation Task"
            "Recommend APIs for the mashup according to its description and give the reason for each recommendation"
            "Mashup Description: {mashup}"
            "Recommended APIs: " + self.API_START_TOKEN 
        )
        

        self.setup_api_mappings(api_repo_path)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            attn_implementation="eager"
        )


        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Apply LoRA if enabled
        if self.use_lora:
            print("Applying LoRA to the model...")
            self.model = get_peft_model(self.model, self.lora_config)
            print("LoRA applied successfully!")
            print(f"Trainable parameters: {self.model.num_parameters()}")
            print(f"Total parameters: {self.model.num_parameters(only_trainable=False)}")
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"Trainable params: {trainable_params:,} || Total params: {total_params:,} || Trainable%: {100 * trainable_params / total_params:.4f}")
        
        self.model.to(self.DEVICE)
        print(self.tokenizer.get_vocab().get("<API_twitter>")) 

    def setup_api_mappings(self, api_repo_path):
        """Setup API special tokens and mappings - exactly matching the first file"""
        self.api_special_tokens = {}
        self.special_token_to_api = {}
        self.api_to_index = {}
        self.index_to_api = {}
        
        # Load API list
        with open(api_repo_path, 'r') as f:
            api_list = json.load(f)
            
        # Setup special tokens for APIs
        api_token_ids_list = []
        for i, api_name in enumerate(api_list):
            # Create special token for API
            api_token = f"<API_{api_name}>"
            self.api_special_tokens[api_name] = api_token
            
            # Add special token to tokenizer if not already added
            if api_token not in self.tokenizer.get_added_vocab():
                special_tokens = {"additional_special_tokens": [api_token]}
                self.tokenizer.add_special_tokens(special_tokens)
                # Resize embeddings after adding special tokens
                if hasattr(self.model, 'resize_token_embeddings'):
                    self.model.resize_token_embeddings(len(self.tokenizer))
            
            # Create mappings
            token_id = self.tokenizer.convert_tokens_to_ids(api_token)
            self.special_token_to_api[api_token] = api_name
            self.api_to_index[api_name] = i
            self.index_to_api[i] = api_name
            api_token_ids_list.append(token_id)
        
        # Store API token IDs as tensor on the correct device
        self.api_token_ids = torch.tensor(api_token_ids_list, dtype=torch.long, device=self.DEVICE)
        self.num_apis = len(api_list)
        print(f"Loaded {self.num_apis} APIs from {api_repo_path}")

    def format_example(self, example):
        """Format example for training - exactly matching the first file"""
        mashup = json.dumps(example["prompt"], ensure_ascii=False)
        apis = example.get("completion", [])
        # reason = example.get("reason", "")
        
        # Convert reason APIs (***api***) to special tokens
        # converted_reason = self.convert_reason_apis_to_tokens(reason)
        
        # Convert API names to special tokens
        if isinstance(apis, list):
            api_tokens = [self.api_special_tokens.get(api, f"<API_{api}>") for api in apis]
        
        text = self.prompt_template.format(
            mashup=mashup, 
            completion=apis,
            # reason=converted_reason
        )
        
        # Reference completion for GRPO
        ref_completion = api_tokens + [self.API_STOP_TOKEN]
        
        return text, ref_completion, apis



    def prepare_dataset(self, raw_ds):
        """Prepare dataset - matching the first file's approach"""
        entries = [self.format_example(i) for i in raw_ds]
        texts, ref_completions, targets = zip(*entries)

        enc = self.tokenizer(
            list(texts),
            return_tensors=None,
            padding=False,
            truncation=True
        )
        
        return Dataset.from_dict({
            "prompt": texts,
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "completion": list(targets),      # 原始 token 序列或 API 列表
            "target_apis": list(targets)     # 纯 API 名称列表
        })

    def ndcg(self, target, pred):
        """Calculate Normalized Discounted Cumulative Gain - Fixed version"""
        if not target or not pred:
            return 0.0
            
        dcg = 0.0
        relevant_count = 0
            
        for i, p in enumerate(pred, 1):
            if p in target:
                relevant_count += 1
                dcg += 1.0 / np.log2(i + 1)
            
        if relevant_count == 0:
            return 0.0
            
        # 关键修改：IDCG 应该基于 min(len(target), relevant_count)
        # 因为理想情况下，我们最多只能预测出 len(target) 个正确项
        ideal_relevant_count = min(len(target), len(pred))
        
        # Calculate IDCG (ideal DCG)
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_relevant_count + 1))
            

        result = dcg / idcg if idcg > 0 else 0.0
        return result

    def mean_average_precision(self, target, pred):
        """Calculate Mean Average Precision - Fixed version"""
        if not pred or not target:
            return 0.0
        
        relevant_items_count = 0
        sum_precision = 0.0
        
        for i, item in enumerate(pred, 1):
            if item in target:
                relevant_items_count += 1
                precision_at_i = relevant_items_count / i
                sum_precision += precision_at_i
        
        if relevant_items_count == 0:
            return 0.0
        

        map_score = sum_precision / len(target) 
        

        return  map_score

    def count_apis_mentioned_in_reason(self, reason_text, recommended_apis):
        """Count how many of the recommended APIs are mentioned in the reason text"""
        if not reason_text or not recommended_apis:
            return 0
        
        mentioned_count = 0
        reason_lower = reason_text.lower()
        
        for api in recommended_apis:
            # Check if API name is mentioned in the reason
            if (api.lower() in reason_lower or 
                self.api_special_tokens.get(api, '').lower() in reason_lower):
                mentioned_count += 1
        
        return mentioned_count
    
    def extract_apis_from_reason(self, reason_text):
        if not reason_text:
            return set()
        
        import re

        api_pattern = r"<API_([^>]+)>"
        matches = re.findall(api_pattern, reason_text)
        
        return set(matches)

    def compute_enhanced_reason_reward(self, recommended_apis, generated_reason):
        # reason(S)
        reason_mentioned_apis = self.extract_apis_from_reason(generated_reason) if generated_reason else set()
        

        R = set(recommended_apis) if recommended_apis else set()
        S = reason_mentioned_apis
        
      
        intersection = R.intersection(S)
        
        # (JS) = MR = |R ∩ S| / |R| if |R| > 0, else 0
        # Reasoning Recall
        if len(R) > 0:
            JS = len(intersection) / len(R)
        else:
            JS = 0.0  
        # Reasoning Precision
        # (SS) = |R ∩ S| / |S| if |S| > 0, else 0
        if len(S) > 0:
            SS = len(intersection) / len(S)
        else:
            SS = 0.0  
        
        reason_reward = 0.5 * JS + 0.5 * SS
        
        return reason_reward

    def compute_reward(self, target_apis, completion):
        api_stop_pos = completion.find(self.API_STOP_TOKEN)

        # Extract reason
        reason = ""
        if self.REASON_START_TOKEN in completion:
            reason_start = completion.find(self.REASON_START_TOKEN) + len(self.REASON_START_TOKEN)
            if self.REASON_STOP_TOKEN in completion:
                reason_end = completion.find(self.REASON_STOP_TOKEN)
                reason = completion[reason_start:reason_end].strip()
            else:
                reason = completion[reason_start:].strip()

        if api_stop_pos > 0:
            api_section = completion[:api_stop_pos]
        else:
            api_section = completion
        

        apis = []

        for api_name, api_token in self.api_special_tokens.items():
            if api_token in api_section:
                apis.append(api_name)


        gen_set = set(apis)
        target_set = set(target_apis)
        
        correct = len(gen_set.intersection(target_set))
        precision = correct / max(1, len(gen_set))
        recall = correct / max(1, len(target_set))
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        ndcg_score = self.ndcg(target_set, gen_set)
        map_score = self.mean_average_precision(target_set, gen_set)
        print(target_set)
        print(gen_set)
        print(reason)
        reason_reward = self.compute_enhanced_reason_reward(gen_set, reason)
        
        base_reward = 0.4*f1 +0.1*recall + 0.1*precision + 0.2*ndcg_score + 0.2*map_score
        # base_reward = 0.25*recall + 0.25*precision + 0.25*ndcg_score + 0.25*map_score
        final_reward = base_reward + reason_reward 
        
        del base_reward,reason_reward
        
        return torch.tensor(final_reward, device=self.DEVICE, dtype=torch.float32)
            
    def find_target_api(self, prompt):
        
        def extract_mashup_from_prompt(prompt):

            start_idx = prompt.find("Mashup Description:")
            if start_idx == -1:
                
                return prompt.strip()
  
            end_idx = prompt.find("Recommended APIs:")
            if end_idx == -1:
                end_idx = len(prompt)
            

            mashup_part = prompt[start_idx + len("Mashup Description:"):end_idx].strip()
            

            if mashup_part.startswith('"') and mashup_part.endswith('"'):
                mashup_part = mashup_part[1:-1]
                
            return mashup_part

        def find_completion_by_mashup(dataset, mashup_description):
            
            for item in dataset:
    
                dataset_prompt = item.get("prompt", "")
                if not dataset_prompt:
                    continue
                    

                if mashup_description in dataset_prompt:

                    return item.get("completion", [])
            
            return None


        mashup = extract_mashup_from_prompt(prompt)
        if not mashup:
            print("error")
            return None
        

        dataset_path = "train_data_reason_ds.jsonl"
        dataset = load_dataset('json', data_files=dataset_path)['train']
        

        completion = find_completion_by_mashup(dataset, mashup)
        
        if not completion:
            completion=[]
            print("error completion")
        
        return completion
    
    def save_model(self, output_dir: str):
        
        os.makedirs(output_dir, exist_ok=True)
        
        if self.use_lora:
            # Save LoRA weights
            self.model.save_pretrained(output_dir)
            print(f"LoRA model saved to {output_dir}")
        else:
            # Save full model
            self.model.save_pretrained(output_dir)
            print(f"Full model saved to {output_dir}")
        
        # Always save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        print(f"Tokenizer saved to {output_dir}")
    
    def train(
        self,
        train_path: str,
        output_dir: str,
        epochs: int = 3,
        lr: float = 5e-6,
        evaluation_steps: int = 500
    ):
        """Train model using GRPO from TRL with LoRA support"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Load training data
        raw = load_dataset("json", data_files=train_path)["train"]
        
        # Prepare dataset
        train_dataset = self.prepare_dataset(raw)
        
        # Create GRPO config with adjusted learning rate for LoRA
        # if self.use_lora:
        #     # LoRA typically uses higher learning rates
        #     adjusted_lr = lr * 10 if lr < 1e-4 else lr
        #     print(f"Using adjusted learning rate for LoRA: {adjusted_lr}")
        # else:
        #     adjusted_lr = lr
        
        grpo_config = GRPOConfig(
            max_completion_length=300,
            gradient_accumulation_steps=1,
            temperature=1.5,
            num_train_epochs=epochs,
            save_steps=5000,
        )

        # Define reward function - matching first file exactly
        def reward_fn(prompts, completions, **kwargs):
            rewards = []
            
            for prompt, completion in zip(prompts, completions):
                target_api=self.find_target_api(prompt)
                r = self.compute_reward( 
                        target_api,
                        completion
                    ).detach().cpu()
                
                rewards.append(r)
            

            torch.cuda.empty_cache()
            gc.collect()
            

            result = torch.stack([r.to(self.DEVICE) for r in rewards])
            print(f"Batch rewards: {result}")
            return result
            
        # Create GRPO trainer
        trainer = GRPOTrainer(
            model=self.model.to(self.DEVICE),
            processing_class=self.tokenizer,
            args=grpo_config,
            train_dataset=train_dataset,
            reward_funcs=reward_fn,
        )
        
        # Start training
        trainer.train()
        
        # Save the model and tokenizer using our custom save method
        self.save_model(output_dir)
        print(f"GRPO training with LoRA complete. Model saved to {output_dir}")

if __name__ == '__main__':
    # Paths to resources
    api_list_file = 'used_api_list.json'
    train_data_file = 'train_data_reason_ds.jsonl'
    output_dir = 'api_grpo_train/phase1'
    base_model = 'api_reason_ds/phase1'

    # Create and train the model with LoRA
    trainer = APIRecommendationGRPOTrainer(
        model_path=base_model,
        api_repo_path=api_list_file,
        use_lora=True,
        lora_r=32,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    trainer.train(
        train_path=train_data_file,
        output_dir=output_dir,
        epochs=10,
        # 20 This is a mistake in the initial version of the paper WAR-Re
        lr=5e-6,
        evaluation_steps=400
    )
