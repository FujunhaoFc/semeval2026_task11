"""
SemEval 2026 Task 11 - Training with LLM Data Generation
DeBERTa-v3-large + LoRA + SCL + LLM Synthetic Data
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import re
import time
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional
from collections import Counter

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    DebertaV2Model,
    get_cosine_schedule_with_warmup
)
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Training configuration with LLM data generation"""
    
    # ==================== LLM Data Generation ====================
    use_llm_generation = True
    
    # Provider 配置 
    # 可选: "siliconflow", "deepseek", "qwen", "openai"
    llm_provider = "siliconflow"
    
    # 模型配置 
    llm_model = "deepseek-ai/DeepSeek-R1"  # 硅基流动上的 R1
    # llm_model = "deepseek-reasoner"       # DeepSeek 官方的 R1
    # llm_model = "qwen-turbo"              # 阿里云百炼
    # llm_model = "gpt-3.5-turbo"           # OpenAI
    
    api_key = None  # 从环境变量读取，或在这里直接设置
    num_generated_samples = 10000  # 总生成样本数
    generation_batch_size = 50  # 批次大小
    
    # API 调用配置
    api_timeout = 60  # 超时时间(秒)
    api_retry = 3     # 重试次数
    api_delay = 0.3   # 请求间隔(秒)，避免限流
    
    # ==================== Quality Control ====================
    min_length = 30   # 最小字符数
    max_length = 500  # 最大字符数
    required_keywords = ["all", "some", "no", "every", "not"]  # 至少包含一个
    conclusion_markers = ["therefore", "thus", "hence", "consequently"]  # 结论标记
    
    # ==================== Model ====================
    model_name = "microsoft/deberta-v3-large"
    max_length = 256
    use_scl = True
    scl_temperature = 0.07
    scl_projection_dim = 256
    
    # ==================== LoRA ====================
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.15
    
    # ==================== Training ====================
    num_epochs = 20
    batch_size = 8
    gradient_accumulation_steps = 2
    learning_rate = 2e-4
    warmup_ratio = 0.1
    weight_decay = 0.02
    scl_loss_weight = 0.5
    max_grad_norm = 1.0
    
    # ==================== Data ====================
    data_path = "train_data/task1/train_data.json"
    val_path = None  # 自定义验证集路径（如果指定，则使用该文件作为完整验证集）
    val_split = 0.15
    generated_data_path = "generated_data.json"
    train_from_generated = False  # 是否只用生成数据训练
    
    # ==================== Output ====================
    output_dir = "outputs_llm_generate"
    save_steps = 200
    
    # ==================== Early Stopping ====================
    early_stopping = True
    early_stopping_patience = 5
    
    # ==================== Misc ====================
    seed = 42
    fp16 = True


# ============================================================================
# Provider Configuration
# ============================================================================

PROVIDER_CONFIG = {
    "siliconflow": {
        "env_key": "SILICONFLOW_API_KEY",
        "base_url": "https://api.siliconflow.cn/v1",
        "default_model": "deepseek-ai/DeepSeek-R1",
        "models": ["deepseek-ai/DeepSeek-R1", "deepseek-ai/DeepSeek-V3", "Qwen/Qwen2.5-72B-Instruct"]
    },
    "deepseek": {
        "env_key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-reasoner",
        "models": ["deepseek-chat", "deepseek-reasoner"]
    },
    "qwen": {
        "env_key": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-turbo",
        "models": ["qwen-turbo", "qwen-plus", "qwen-max"]
    },
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-3.5-turbo",
        "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
    }
}


# ============================================================================
# Prompts for Syllogism Generation (优化版 - 针对逻辑推理)
# ============================================================================

PROMPTS = {
    "valid_plausible": """You are a logic expert. Generate a VALID and PLAUSIBLE categorical syllogism.

VALIDITY means the conclusion MUST follow necessarily from the premises by the rules of syllogistic logic.
PLAUSIBILITY means the content uses realistic, believable entities and facts.

IMPORTANT: Be CREATIVE and use DIVERSE topics! Avoid common examples like birds/mammals/dogs.
Choose from varied domains: professions, foods, places, vehicles, sports, music, science, history, etc.

Valid syllogism forms include:
- Barbara (AAA-1): All M are P. All S are M. Therefore, all S are P.
- Celarent (EAE-1): No M are P. All S are M. Therefore, no S are P.
- Darii (AII-1): All M are P. Some S are M. Therefore, some S are P.

Requirements:
- Use quantifiers: all, some, no, every
- Include conclusion marker: therefore, thus, hence
- Format: "Premise 1. Premise 2. Therefore, Conclusion."
- BE CREATIVE with entities - avoid birds, mammals, dogs, cats!

Generate exactly 1 UNIQUE syllogism. Output ONLY the syllogism, nothing else.""",

    "valid_implausible": """You are a logic expert. Generate a VALID but IMPLAUSIBLE categorical syllogism.

VALIDITY means the conclusion MUST follow necessarily from the premises (correct logical form).
IMPLAUSIBILITY means the content is absurd or fantastical (but the logic is still correct).

IMPORTANT: Be HIGHLY CREATIVE! Use surreal combinations like:
- Abstract concepts as physical objects (thoughts that swim, ideas that dance)
- Impossible properties (silent thunderstorms, frozen flames)
- Category violations (emotions as furniture, numbers as animals)

Requirements:
- The logical structure must be VALID
- Use wildly creative, absurd entities - be imaginative!
- Use quantifiers: all, some, no, every
- Include conclusion marker: therefore, thus, hence

Generate exactly 1 UNIQUE syllogism. Output ONLY the syllogism, nothing else.""",

    "invalid_plausible": """You are a logic expert. Generate an INVALID but PLAUSIBLE categorical syllogism.

INVALIDITY means the conclusion does NOT logically follow from the premises (contains a logical fallacy).
PLAUSIBILITY means the content sounds reasonable and uses realistic entities.

IMPORTANT: Use DIVERSE realistic topics! Avoid birds/mammals. Try:
- Professions and skills
- Geographic locations
- Food and cooking
- Technology and devices
- Historical events

Common fallacies to use:
- Undistributed middle: All A are B. All C are B. Therefore, all A are C. (WRONG!)
- Illicit major: All A are B. No C are A. Therefore, no C are B. (WRONG!)
- Affirming the consequent: All A are B. X is B. Therefore, X is A. (WRONG!)

Requirements:
- The conclusion must NOT logically follow
- Use realistic but VARIED entities
- Use quantifiers: all, some, no, every
- Include conclusion marker: therefore, thus, hence

Generate exactly 1 UNIQUE syllogism. Output ONLY the syllogism, nothing else.""",

    "invalid_implausible": """You are a logic expert. Generate an INVALID and IMPLAUSIBLE categorical syllogism.

INVALIDITY means the conclusion does NOT logically follow (contains a logical fallacy).
IMPLAUSIBILITY means the content is absurd or fantastical.

IMPORTANT: Be WILDLY CREATIVE with absurd content! Examples:
- "All whispers are purple elephants"
- "Some mathematical equations are jealous"
- "No Tuesday afternoons are edible"

Requirements:
- Use a logical fallacy (undistributed middle, illicit major/minor, etc.)
- Use HIGHLY creative, surreal entities
- Use quantifiers: all, some, no, every
- Include conclusion marker: therefore, thus, hence

Generate exactly 1 UNIQUE syllogism. Output ONLY the syllogism, nothing else."""
}


# ============================================================================
# LLM Data Generator (支持多个免费 API)
# ============================================================================

class LLMDataGenerator:
    """Generate syllogism data using multiple LLM APIs"""
    
    def __init__(self, config: Config):
        self.config = config
        self.provider = config.llm_provider
        
        if self.provider not in PROVIDER_CONFIG:
            raise ValueError(f"Unknown provider: {self.provider}. Choose from: {list(PROVIDER_CONFIG.keys())}")
        
        provider_cfg = PROVIDER_CONFIG[self.provider]
        
        # 获取 API Key
        self.api_key = config.api_key or os.getenv(provider_cfg["env_key"])
        if not self.api_key:
            raise ValueError(
                f"API key not found!\n"
                f"Set environment variable: export {provider_cfg['env_key']}='your-key'\n"
                f"Or pass --api_key argument"
            )
        
        self.base_url = provider_cfg["base_url"]
        self.model = config.llm_model or provider_cfg["default_model"]
        
        print(f"\n{'='*60}")
        print(f"[LLM Generator] Initialized")
        print(f"  Provider: {self.provider}")
        print(f"  Model: {self.model}")
        print(f"  Base URL: {self.base_url}")
        print(f"{'='*60}")
    
    def call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM API using OpenAI-compatible format"""
        try:
            from openai import OpenAI
            
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.config.api_timeout
            )
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,  # 提高到1.0增加多样性
                max_tokens=300
            )
            
            content = response.choices[0].message.content.strip()
            
            # 处理 DeepSeek R1 的思考过程 (如果有)
            # R1 可能返回 <think>...</think> 标签包裹的思考过程
            if "<think>" in content and "</think>" in content:
                # 提取思考过程之后的实际回答
                parts = content.split("</think>")
                if len(parts) > 1:
                    content = parts[-1].strip()
            
            return content
            
        except Exception as e:
            print(f"  API call failed: {e}")
            return None
    
    def validate_sample(self, syllogism: str) -> bool:
        """Quality check for generated sample"""
        if not syllogism:
            return False
        
        # 长度检查
        if len(syllogism) < self.config.min_length or len(syllogism) > self.config.max_length:
            return False
        
        syllogism_lower = syllogism.lower()
        
        # 必须包含至少一个量词
        if not any(kw in syllogism_lower for kw in self.config.required_keywords):
            return False
        
        # 必须包含结论标记
        if not any(marker in syllogism_lower for marker in self.config.conclusion_markers):
            return False
        
        # 结构检查：至少2个句点（两个前提 + 结论）
        if syllogism.count('.') < 2:
            return False
        
        # 排除明显的非三段论内容
        bad_patterns = [
            "i cannot", "i can't", "sorry", "as an ai",
            "here is", "here's", "example:", "note:"
        ]
        if any(p in syllogism_lower for p in bad_patterns):
            return False
        
        return True
    
    def normalize_text(self, text: str) -> str:
        """标准化文本用于去重比较"""
        # 去除引号、多余空格、统一小写
        text = text.replace('"', '').replace("'", "")
        text = ' '.join(text.lower().split())
        return text
    
    def is_duplicate(self, syllogism: str, existing: set) -> bool:
        """检查是否重复 - 只检查完全匹配"""
        normalized = self.normalize_text(syllogism)
        return normalized in existing
    
    def generate_samples(self, num_samples: int) -> List[Dict]:
        """Generate synthetic syllogism samples with deduplication"""
        print(f"\n[Data Generation] Generating {num_samples} samples...")
        print(f"  Using model: {self.model}")
        print(f"  Deduplication: ENABLED")
        
        generated = []
        seen_texts = set()  # 用于去重
        samples_per_type = num_samples // 4
        
        types = [
            ("valid_plausible", True, True),
            ("valid_implausible", True, False),
            ("invalid_plausible", False, True),
            ("invalid_implausible", False, False)
        ]
        
        total_attempts = 0
        total_success = 0
        total_duplicates = 0
        
        for prompt_type, validity, plausibility in types:
            print(f"\n  Generating {samples_per_type} {prompt_type} samples...")
            prompt = PROMPTS[prompt_type]
            
            pbar = tqdm(range(samples_per_type), desc=f"  {prompt_type}")
            attempts = 0
            successful = 0
            duplicates = 0
            max_attempts = samples_per_type * 10  # 增加尝试次数因为有去重
            
            while successful < samples_per_type and attempts < max_attempts:
                attempts += 1
                total_attempts += 1
                
                # 调用 LLM
                syllogism = self.call_llm(prompt)
                
                if syllogism and self.validate_sample(syllogism):
                    # 去重检查
                    if self.is_duplicate(syllogism, seen_texts):
                        duplicates += 1
                        total_duplicates += 1
                        continue
                    
                    # 添加到已见集合
                    seen_texts.add(self.normalize_text(syllogism))
                    
                    sample = {
                        "id": f"llm_{prompt_type}_{successful}",
                        "syllogism": syllogism,
                        "validity": validity,
                        "plausibility": plausibility
                    }
                    generated.append(sample)
                    successful += 1
                    total_success += 1
                    pbar.update(1)
                
                # Rate limiting
                time.sleep(self.config.api_delay)
            
            pbar.close()
            success_rate = successful / attempts * 100 if attempts > 0 else 0
            print(f"  ✓ Generated {successful}/{samples_per_type} (success rate: {success_rate:.1f}%, duplicates filtered: {duplicates})")
        
        overall_rate = total_success / total_attempts * 100 if total_attempts > 0 else 0
        print(f"\n[Data Generation] Complete!")
        print(f"  Total generated: {len(generated)} unique samples")
        print(f"  Total duplicates filtered: {total_duplicates}")
        print(f"  Overall success rate: {overall_rate:.1f}%")
        
        return generated


# ============================================================================
# Dataset
# ============================================================================

class SyllogismDataset(Dataset):
    """Dataset with flexible train/val configuration
    
    Supports multiple modes:
    1. Default: Split original data into train/val, optionally add generated to train
    2. Custom val: Use specified file as validation set
    3. Train from generated: Use only generated data for training
    """
    
    def __init__(self, data_path: str, tokenizer, config: Config, split='train', 
                 generated_data: Optional[List[Dict]] = None,
                 val_data_path: Optional[str] = None,
                 train_from_generated: bool = False):
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        
        # ==================== Mode 1: Train from generated only ====================
        if train_from_generated and split == 'train':
            if generated_data is None:
                raise ValueError("train_from_generated requires generated_data")
            self.data = generated_data.copy()
            print(f"\n[Dataset] Mode: Train from GENERATED data only")
            print(f"  Loaded {len(self.data)} generated samples for training")
        
        # ==================== Mode 2: Custom validation set ====================
        elif val_data_path and split == 'val':
            with open(val_data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"\n[Dataset] Mode: Custom validation set")
            print(f"  Loaded {len(self.data)} samples from {val_data_path}")
        
        # ==================== Mode 3: Default (split original data) ====================
        else:
            with open(data_path, 'r', encoding='utf-8') as f:
                original_data = json.load(f)
            
            print(f"\n[Dataset] Mode: Default (split original data)")
            print(f"  Original data: {len(original_data)} samples")
            
            # Split original data
            train_data, val_data = train_test_split(
                original_data,
                test_size=config.val_split,
                random_state=config.seed,
                stratify=[d['validity'] for d in original_data]
            )
            
            if split == 'train':
                self.data = train_data.copy()
                # Add generated data to training set
                if generated_data and not train_from_generated:
                    self.data.extend(generated_data)
                    print(f"  Added {len(generated_data)} LLM-generated samples to training")
            else:
                self.data = val_data
        
        # ==================== Group by validity for SCL ====================
        self.valid_indices = [i for i, item in enumerate(self.data) if item['validity']]
        self.invalid_indices = [i for i, item in enumerate(self.data) if not item['validity']]
        
        print(f"[Dataset] Final {split} set: {len(self.data)} samples")
        print(f"  Valid: {len(self.valid_indices)}, Invalid: {len(self.invalid_indices)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['syllogism']
        label = 1 if item['validity'] else 0
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
        }
        
        # SCL pairs (train only)
        if self.split == 'train' and self.config.use_scl:
            # Positive (same validity)
            pos_pool = self.valid_indices if label == 1 else self.invalid_indices
            if len(pos_pool) > 1:
                pos_idx = random.choice([i for i in pos_pool if i != idx])
                pos_text = self.data[pos_idx]['syllogism']
            else:
                pos_text = text
            
            # Negative (different validity)
            neg_pool = self.invalid_indices if label == 1 else self.valid_indices
            if len(neg_pool) > 0:
                neg_idx = random.choice(neg_pool)
                neg_text = self.data[neg_idx]['syllogism']
            else:
                neg_text = text
            
            # Tokenize pairs
            pos_enc = self.tokenizer(
                pos_text,
                max_length=self.config.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            neg_enc = self.tokenizer(
                neg_text,
                max_length=self.config.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            result['positive_input_ids'] = pos_enc['input_ids'].squeeze(0)
            result['positive_attention_mask'] = pos_enc['attention_mask'].squeeze(0)
            result['negative_input_ids'] = neg_enc['input_ids'].squeeze(0)
            result['negative_attention_mask'] = neg_enc['attention_mask'].squeeze(0)
        
        return result


# ============================================================================
# Model
# ============================================================================

class DeBertaSCL(nn.Module):
    """DeBERTa with Supervised Contrastive Learning"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load DeBERTa
        self.deberta = DebertaV2Model.from_pretrained(config.model_name)
        hidden_size = self.deberta.config.hidden_size
        
        # Apply LoRA
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["query_proj", "value_proj", "key_proj", "dense"],
            bias="none",
        )
        self.deberta = get_peft_model(self.deberta, peft_config)
        
        # Classification head
        self.classifier = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(0.1)
        
        # Projection head for SCL
        if config.use_scl:
            self.projection_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size, config.scl_projection_dim)
            )
        
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask, labels=None,
                positive_input_ids=None, positive_attention_mask=None,
                negative_input_ids=None, negative_attention_mask=None):
        
        # Get embeddings (anchor)
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        
        # Classification
        logits = self.classifier(pooled)
        
        loss = None
        if labels is not None:
            # Classification loss
            ce_loss = self.criterion(logits, labels)
            loss = ce_loss
            
            # SCL loss
            if self.config.use_scl and positive_input_ids is not None:
                # Encode positive
                pos_outputs = self.deberta(input_ids=positive_input_ids,
                                          attention_mask=positive_attention_mask)
                pos_pooled = pos_outputs.last_hidden_state[:, 0, :]
                
                # Encode negative
                neg_outputs = self.deberta(input_ids=negative_input_ids,
                                          attention_mask=negative_attention_mask)
                neg_pooled = neg_outputs.last_hidden_state[:, 0, :]
                
                # Project
                anchor_proj = F.normalize(self.projection_head(pooled), dim=1)
                pos_proj = F.normalize(self.projection_head(pos_pooled), dim=1)
                neg_proj = F.normalize(self.projection_head(neg_pooled), dim=1)
                
                # Similarities
                pos_sim = torch.sum(anchor_proj * pos_proj, dim=1) / self.config.scl_temperature
                neg_sim = torch.sum(anchor_proj * neg_proj, dim=1) / self.config.scl_temperature
                
                # InfoNCE loss
                contrastive_logits = torch.stack([pos_sim, neg_sim], dim=1)
                contrastive_labels = torch.zeros(len(labels), dtype=torch.long, device=labels.device)
                scl_loss = F.cross_entropy(contrastive_logits, contrastive_labels)
                
                # Combined loss
                loss = (1 - self.config.scl_loss_weight) * ce_loss + \
                       self.config.scl_loss_weight * scl_loss
        
        return {'loss': loss, 'logits': logits}


# ============================================================================
# Training Functions
# ============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, optimizer, scheduler, device, config):
    model.train()
    total_loss = 0
    predictions = []
    labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    optimizer.zero_grad()
    
    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        batch_labels = batch['labels'].to(device)
        
        kwargs = {}
        if 'positive_input_ids' in batch:
            kwargs['positive_input_ids'] = batch['positive_input_ids'].to(device)
            kwargs['positive_attention_mask'] = batch['positive_attention_mask'].to(device)
            kwargs['negative_input_ids'] = batch['negative_input_ids'].to(device)
            kwargs['negative_attention_mask'] = batch['negative_attention_mask'].to(device)
        
        # Forward
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=batch_labels,
            **kwargs
        )
        
        loss = outputs['loss'] / config.gradient_accumulation_steps
        loss.backward()
        
        if (step + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * config.gradient_accumulation_steps
        preds = torch.argmax(outputs['logits'], dim=1)
        predictions.extend(preds.cpu().numpy())
        labels.extend(batch_labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item() * config.gradient_accumulation_steps})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(labels, predictions)
    
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    predictions = []
    labels = []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        batch_labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=batch_labels
        )
        
        total_loss += outputs['loss'].item()
        preds = torch.argmax(outputs['logits'], dim=1)
        predictions.extend(preds.cpu().numpy())
        labels.extend(batch_labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(labels, predictions)
    
    return avg_loss, accuracy


# ============================================================================
# Main
# ============================================================================

def main(config):
    set_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(config.output_dir, exist_ok=True)
    
    print("="*70)
    print("SemEval 2026 Task 11 - Training with LLM Data Generation (v4)")
    print("="*70)
    print(f"Device: {device}")
    print(f"Model: {config.model_name}")
    print(f"LLM Generation: {config.use_llm_generation}")
    print(f"Train from generated only: {config.train_from_generated}")
    print(f"Custom validation path: {config.val_path}")
    if config.use_llm_generation:
        print(f"  Provider: {config.llm_provider}")
        print(f"  LLM Model: {config.llm_model}")
        print(f"  Target samples: {config.num_generated_samples}")
    print("="*70)
    
    # ==================== Generate or Load LLM Data ====================
    generated_data = None
    if config.use_llm_generation or config.train_from_generated:
        generated_cache = Path(config.generated_data_path)
        
        if generated_cache.exists():
            print(f"\n[Cache] Loading generated data from {generated_cache}")
            with open(generated_cache, 'r', encoding='utf-8') as f:
                generated_data = json.load(f)
            print(f"[Cache] Loaded {len(generated_data)} samples")
            
            # 显示数据分布
            valid_count = sum(1 for d in generated_data if d['validity'])
            print(f"  Valid: {valid_count}, Invalid: {len(generated_data) - valid_count}")
        else:
            if config.train_from_generated:
                raise FileNotFoundError(
                    f"Generated data file not found: {generated_cache}\n"
                    f"Please generate data first or provide correct path with --generated_data_path"
                )
            
            print("\n[Generation] Generating new data with LLM...")
            generator = LLMDataGenerator(config)
            generated_data = generator.generate_samples(config.num_generated_samples)
            
            # Save generated data
            with open(generated_cache, 'w', encoding='utf-8') as f:
                json.dump(generated_data, f, indent=2, ensure_ascii=False)
            print(f"\n[Cache] Saved generated data to {generated_cache}")
    
    # ==================== Tokenizer ====================
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # ==================== Datasets ====================
    print("[2/5] Loading datasets...")
    
    train_dataset = SyllogismDataset(
        config.data_path,
        tokenizer,
        config,
        split='train',
        generated_data=generated_data,
        train_from_generated=config.train_from_generated
    )
    
    val_dataset = SyllogismDataset(
        config.data_path,
        tokenizer,
        config,
        split='val',
        generated_data=None,
        val_data_path=config.val_path
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=2
    )
    
    # ==================== Model ====================
    print("[3/5] Creating model...")
    model = DeBertaSCL(config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total: {total_params:,} | Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # ==================== Optimizer ====================
    print("[4/5] Setting up optimizer...")
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    num_training_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # ==================== Training ====================
    print("[5/5] Training...")
    print("="*70)
    
    best_val_acc = 0
    early_stopping_counter = 0
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, config
        )
        
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stopping_counter = 0
            
            save_path = Path(config.output_dir) / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'config': config
            }, save_path)
            print(f"  ✓ Saved best model (acc: {val_acc:.4f})")
        else:
            early_stopping_counter += 1
            print(f"  Early stopping counter: {early_stopping_counter}/{config.early_stopping_patience}")
        
        # Early stopping
        if config.early_stopping and early_stopping_counter >= config.early_stopping_patience:
            print(f"\n⚠ Early stopping triggered at epoch {epoch + 1}")
            break
    
    # ==================== Summary ====================
    print("\n" + "="*70)
    print(f"Training completed!")
    print(f"  Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"  Model saved to: {config.output_dir}/best_model.pt")
    print("="*70)
    
    # Save summary
    summary_path = Path(config.output_dir) / "training_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"SemEval 2026 Task 11 Training Summary (v4)\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Model: {config.model_name}\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")
        f.write(f"Total Epochs: {epoch + 1}\n")
        f.write(f"Training Samples: {len(train_dataset)}\n")
        f.write(f"Validation Samples: {len(val_dataset)}\n")
        f.write(f"\nExperiment Mode:\n")
        f.write(f"  Train from generated only: {config.train_from_generated}\n")
        f.write(f"  Custom validation path: {config.val_path}\n")
        if generated_data:
            f.write(f"\nLLM Data:\n")
            f.write(f"  Generated Samples: {len(generated_data)}\n")
            f.write(f"  Provider: {config.llm_provider}\n")
            f.write(f"  Model: {config.llm_model}\n")
        f.write(f"\nConfiguration:\n")
        f.write(f"  - SCL: {config.use_scl}\n")
        f.write(f"  - LoRA rank: {config.lora_r}\n")
        f.write(f"  - Learning rate: {config.learning_rate}\n")
        f.write(f"  - Batch size: {config.batch_size * config.gradient_accumulation_steps}\n")
    
    print(f"Summary saved to: {summary_path}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SemEval 2026 Task 11 Training with LLM Data Generation (v4)")
    
    # LLM Generation
    parser.add_argument("--no_llm", action="store_true", help="Disable LLM generation (use cached data)")
    parser.add_argument("--llm_samples", type=int, default=10000, help="Number of samples to generate")
    parser.add_argument("--provider", type=str, default="siliconflow",
                       choices=["siliconflow", "deepseek", "qwen", "openai"],
                       help="LLM API provider (default: siliconflow)")
    parser.add_argument("--model", type=str, default=None,
                       help="LLM model name (uses provider default if not specified)")
    parser.add_argument("--api_key", type=str, default=None, help="API key")
    
    # Training
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    
    # Data - NEW OPTIONS
    parser.add_argument("--data_path", type=str, default=None, help="Path to original training data")
    parser.add_argument("--val_path", type=str, default=None, 
                       help="Custom validation data path (use entire file as validation set)")
    parser.add_argument("--generated_data_path", type=str, default=None,
                       help="Path to generated data JSON file")
    parser.add_argument("--train_from_generated", action="store_true",
                       help="Train ONLY from generated data (no original data in training)")
    
    # Output
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    
    args = parser.parse_args()
    
    # Build config
    config = Config()
    
    # LLM settings
    if args.no_llm:
        config.use_llm_generation = False
    if args.llm_samples:
        config.num_generated_samples = args.llm_samples
    if args.provider:
        config.llm_provider = args.provider
        # 设置对应的默认模型
        if args.model is None:
            config.llm_model = PROVIDER_CONFIG[args.provider]["default_model"]
    if args.model:
        config.llm_model = args.model
    if args.api_key:
        config.api_key = args.api_key
    
    # Training settings
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    
    # Data settings - NEW
    if args.data_path:
        config.data_path = args.data_path
    if args.val_path:
        config.val_path = args.val_path
    if args.generated_data_path:
        config.generated_data_path = args.generated_data_path
    if args.train_from_generated:
        config.train_from_generated = True
        # 如果只用生成数据训练，默认关闭新生成
        config.use_llm_generation = False
    
    # Output settings
    if args.output_dir:
        config.output_dir = args.output_dir
    
    main(config)