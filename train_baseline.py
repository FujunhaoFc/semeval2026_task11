"""
SemEval 2026 Task 11 - Optimized Training Script
DeBERTa-v3-large + Data Augmentation + Synthetic Data + SCL

Based on successful 96.53% validation accuracy run
Optimized for stability and preventing data loss
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
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
from collections import Counter

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    DebertaV2Model,  # Use Model, not ForSequenceClassification
    get_cosine_schedule_with_warmup
)
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Optimized configuration based on 96.53% validation accuracy run"""
    
    # Model - DeBERTa-v3-large (memory efficient, proven effective)
    model_name = "microsoft/deberta-v3-large"
    max_length = 256
    use_scl = True
    scl_temperature = 0.07
    scl_projection_dim = 256
    
    # LoRA - Enhanced regularization to prevent overfitting
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.15  # Increased from 0.1
    
    # Training - Optimized based on previous run
    num_epochs = 12  # Reduced from 15 (converged by epoch 10-11)
    batch_size = 4
    gradient_accumulation_steps = 4  # Effective batch size = 16
    learning_rate = 3e-4
    warmup_ratio = 0.1
    weight_decay = 0.02  # Increased from 0.01 for better regularization
    scl_loss_weight = 0.5
    max_grad_norm = 1.0
    
    # Data Augmentation - Keep successful settings
    use_augmentation = True
    augmentation_multiplier = 3  # 960 → 2880 samples
    paraphrase_prob = 0.5
    swap_premises_prob = 0.3
    synonym_replace_prob = 0.3
    
    # Synthetic Data Generation
    generate_synthetic = True
    synthetic_samples = 500
    
    # Data Split
    data_path = "train_data/task1/train_data.json"
    train_split = 0.85
    val_split = 0.15
    
    # Output - PERMANENT STORAGE (critical fix)
    output_dir = "outputs"
    save_steps = 200
    eval_steps = 100
    
    # Early Stopping (prevent overfitting)
    early_stopping = True
    early_stopping_patience = 3
    
    # Regularization
    use_label_smoothing = True
    label_smoothing = 0.1
    
    # Misc
    seed = 42
    fp16 = True


# ============================================================================
# Data Augmentation
# ============================================================================

class SyllogismAugmenter:
    """Advanced data augmentation for syllogisms"""
    
    def __init__(self, config):
        self.config = config
        
        # Quantifier synonyms
        self.quantifier_map = {
            'all': ['every', 'each', 'any'],
            'every': ['all', 'each', 'any'],
            'some': ['certain', 'a few', 'several'],
            'no': ['not any', 'not a single', 'none of the'],
            'not all': ['not every', 'some are not'],
        }
        
        # Logical connectors
        self.connectors = {
            'therefore': ['thus', 'hence', 'consequently', 'it follows that'],
            'thus': ['therefore', 'hence', 'consequently'],
            'hence': ['therefore', 'thus', 'consequently'],
        }
    
    def augment(self, text: str) -> str:
        """Apply multiple augmentation strategies"""
        
        # Strategy 1: Swap premises (30% chance)
        if random.random() < self.config.swap_premises_prob:
            text = self._swap_premises(text)
        
        # Strategy 2: Paraphrase quantifiers (50% chance)
        if random.random() < self.config.paraphrase_prob:
            text = self._paraphrase_quantifiers(text)
        
        # Strategy 3: Paraphrase connectors
        if random.random() < 0.3:
            text = self._paraphrase_connectors(text)
        
        # Strategy 4: Synonym replacement (30% chance)
        if random.random() < self.config.synonym_replace_prob:
            text = self._synonym_replace(text)
        
        return text
    
    def _swap_premises(self, text: str) -> str:
        """Swap premise order"""
        conclusion_markers = ['therefore', 'thus', 'hence', 'consequently']
        
        for marker in conclusion_markers:
            if marker in text.lower():
                parts = re.split(f'\\.\\s*(?={marker})', text, flags=re.IGNORECASE)
                if len(parts) >= 2:
                    premises = parts[0].split('.')
                    if len(premises) >= 2:
                        premises[0], premises[1] = premises[1].strip(), premises[0].strip()
                        return '. '.join(premises) + '.' + parts[1]
        
        return text
    
    def _paraphrase_quantifiers(self, text: str) -> str:
        """Replace quantifiers with synonyms"""
        for original, replacements in self.quantifier_map.items():
            pattern = r'\b' + re.escape(original) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                replacement = random.choice(replacements)
                text = re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)
                break
        return text
    
    def _paraphrase_connectors(self, text: str) -> str:
        """Replace logical connectors"""
        for original, replacements in self.connectors.items():
            pattern = r'\b' + re.escape(original) + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                replacement = random.choice(replacements)
                text = re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)
                break
        return text
    
    def _synonym_replace(self, text: str) -> str:
        """Simple synonym replacement"""
        synonyms = {
            'creature': ['being', 'organism', 'entity'],
            'classified': ['categorized', 'grouped', 'identified'],
            'belongs': ['falls into', 'is part of', 'is included in'],
            'certain': ['specific', 'particular', 'some'],
        }
        
        for word, replacements in synonyms.items():
            if word in text.lower():
                replacement = random.choice(replacements)
                text = re.sub(r'\b' + word + r'\b', replacement, text, count=1, flags=re.IGNORECASE)
                break
        
        return text


# ============================================================================
# Synthetic Data Generation
# ============================================================================

class SyntheticDataGenerator:
    """Generate synthetic syllogisms based on templates"""
    
    def __init__(self):
        # Entity categories
        self.animals = ['dogs', 'cats', 'birds', 'fish', 'horses', 'elephants', 'lions', 'tigers', 'wolves']
        self.plants = ['trees', 'flowers', 'grass', 'ferns', 'moss', 'algae', 'shrubs']
        self.objects = ['chairs', 'tables', 'books', 'pens', 'cars', 'bicycles', 'phones', 'computers']
        self.people = ['teachers', 'students', 'doctors', 'engineers', 'artists', 'musicians', 'athletes']
        self.abstracts = ['ideas', 'concepts', 'thoughts', 'emotions', 'beliefs', 'theories', 'principles']
        
        # Superclass for each category
        self.superclasses = {
            'animals': 'living beings',
            'plants': 'organisms',
            'objects': 'items',
            'people': 'humans',
            'abstracts': 'notions'
        }
    
    def generate(self, num_samples: int) -> List[Dict]:
        """Generate synthetic syllogisms"""
        synthetic_data = []
        
        for i in range(num_samples):
            validity = random.choice([True, False])
            plausibility = random.choice([True, False])
            
            if plausibility:
                syllogism = self._generate_plausible(validity)
            else:
                syllogism = self._generate_implausible(validity)
            
            synthetic_data.append({
                'id': f'synthetic_{i}',
                'syllogism': syllogism,
                'validity': validity,
                'plausibility': plausibility
            })
        
        return synthetic_data
    
    def _generate_plausible(self, validity: bool) -> str:
        """Generate plausible syllogism"""
        category = random.choice(['animals', 'plants', 'objects', 'people'])
        entities = getattr(self, category)
        superclass = self.superclasses[category]
        
        A = random.choice(entities)
        B = random.choice([e for e in entities if e != A])
        C = superclass
        
        if validity:
            return f"All {A} are {B}. All {B} are {C}. Therefore, all {A} are {C}."
        else:
            return f"All {A} are {C}. All {B} are {C}. Therefore, all {A} are {B}."
    
    def _generate_implausible(self, validity: bool) -> str:
        """Generate implausible syllogism"""
        cat1 = random.choice(['animals', 'objects'])
        cat2 = random.choice(['abstracts', 'people'])
        
        A = random.choice(getattr(self, cat1))
        B = random.choice(getattr(self, cat2))
        C = random.choice(['machines', 'tools', 'structures'])
        
        if validity:
            return f"All {A} are {B}. No {B} are {C}. Therefore, no {A} are {C}."
        else:
            return f"Some {A} are {B}. All {C} are {B}. Therefore, all {C} are {A}."


# ============================================================================
# Dataset
# ============================================================================

class EnhancedSyllogismDataset(Dataset):
    """Enhanced dataset with augmentation and synthetic data"""
    
    def __init__(self, data_path, tokenizer, config, split='train'):
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        
        # Load original data
        with open(data_path, 'r') as f:
            original_data = json.load(f)
        
        print(f"\n[Dataset] Original data: {len(original_data)} samples")
        
        # Split data
        if split in ['train', 'val']:
            train_data, val_data = train_test_split(
                original_data,
                test_size=config.val_split,
                random_state=config.seed,
                stratify=[d['validity'] for d in original_data]
            )
            self.data = train_data if split == 'train' else val_data
        else:
            self.data = original_data
        
        # Augmentation (train only)
        if split == 'train' and config.use_augmentation:
            self.augmenter = SyllogismAugmenter(config)
            augmented = []
            
            for _ in range(config.augmentation_multiplier - 1):
                for item in self.data:
                    aug_item = item.copy()
                    aug_item['syllogism'] = self.augmenter.augment(item['syllogism'])
                    aug_item['id'] = f"{item['id']}_aug_{len(augmented)}"
                    augmented.append(aug_item)
            
            self.data.extend(augmented)
            print(f"[Dataset] After augmentation: {len(self.data)} samples")
        
        # Synthetic data (train only)
        if split == 'train' and config.generate_synthetic:
            generator = SyntheticDataGenerator()
            synthetic = generator.generate(config.synthetic_samples)
            self.data.extend(synthetic)
            print(f"[Dataset] After synthetic: {len(self.data)} samples")
        
        # Group by validity for SCL
        self.valid_indices = [i for i, item in enumerate(self.data) if item['validity']]
        self.invalid_indices = [i for i, item in enumerate(self.data) if not item['validity']]
        
        print(f"[Dataset] Final {split}: {len(self.data)} samples")
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
            # Positive
            pos_pool = self.valid_indices if label == 1 else self.invalid_indices
            pos_idx = random.choice([i for i in pos_pool if i != idx])
            pos_text = self.data[pos_idx]['syllogism']
            
            # Negative
            neg_pool = self.invalid_indices if label == 1 else self.valid_indices
            neg_idx = random.choice(neg_pool)
            neg_text = self.data[neg_idx]['syllogism']
            
            # Tokenize
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

class DeBertaSCLModel(nn.Module):
    """DeBERTa-v3-large with SCL and regularization"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load DeBERTa base model (not ForSequenceClassification)
        self.deberta = DebertaV2Model.from_pretrained(config.model_name)
        hidden_size = self.deberta.config.hidden_size
        
        # Apply LoRA
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,  # Changed from SEQ_CLS
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
        
        # Label smoothing
        if config.use_label_smoothing:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, input_ids, attention_mask, labels=None,
                positive_input_ids=None, positive_attention_mask=None,
                negative_input_ids=None, negative_attention_mask=None):
        
        # Get embeddings (anchor)
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
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
                
                # InfoNCE
                contrastive_logits = torch.stack([pos_sim, neg_sim], dim=1)
                contrastive_labels = torch.zeros(len(labels), dtype=torch.long, device=labels.device)
                scl_loss = F.cross_entropy(contrastive_logits, contrastive_labels)
                
                # Combined loss
                loss = (1 - self.config.scl_loss_weight) * ce_loss + \
                       self.config.scl_loss_weight * scl_loss
        
        return {'loss': loss, 'logits': logits}


# ============================================================================
# Training
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
        
        # Backward
        loss.backward()
        
        # Update every N steps
        if (step + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Metrics
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


def main(config):
    set_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    print("="*70)
    print("SemEval 2026 Task 11 - Optimized Training")
    print("="*70)
    print(f"Device: {device}")
    print(f"Model: {config.model_name}")
    print(f"SCL: {config.use_scl}")
    print(f"Augmentation: {config.use_augmentation} (x{config.augmentation_multiplier})")
    print(f"Synthetic Data: {config.generate_synthetic} ({config.synthetic_samples} samples)")
    print(f"Output: {config.output_dir}")
    print(f"Early Stopping: {config.early_stopping} (patience={config.early_stopping_patience})")
    print("="*70)
    
    # Tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Datasets
    print("[2/5] Loading datasets...")
    train_dataset = EnhancedSyllogismDataset(
        config.data_path,
        tokenizer,
        config,
        split='train'
    )
    
    val_dataset = EnhancedSyllogismDataset(
        config.data_path,
        tokenizer,
        config,
        split='val'
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
    
    # Model
    print("[3/5] Creating model...")
    model = DeBertaSCLModel(config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total: {total_params:,} | Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Optimizer
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
    
    # Training
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
                'val_loss': val_loss,
                'config': config
            }, save_path)
            print(f"  ✓ Saved best model (acc: {val_acc:.4f})")
        else:
            early_stopping_counter += 1
            print(f"  Early stopping counter: {early_stopping_counter}/{config.early_stopping_patience}")
        
        # Early stopping
        if config.early_stopping and early_stopping_counter >= config.early_stopping_patience:
            print(f"\n Early stopping triggered at epoch {epoch + 1}")
            break
    
    print("\n" + "="*70)
    print(f"Training completed! Best Val Acc: {best_val_acc:.4f}")
    print(f"Model saved to: {config.output_dir}/best_model.pt")
    print("="*70)
    
    # Save training summary
    summary_path = Path(config.output_dir) / "training_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"SemEval 2026 Task 11 Training Summary\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Model: {config.model_name}\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")
        f.write(f"Total Epochs: {epoch + 1}\n")
        f.write(f"Training Samples: {len(train_dataset)}\n")
        f.write(f"Validation Samples: {len(val_dataset)}\n")
        f.write(f"\nConfiguration:\n")
        f.write(f"  - SCL: {config.use_scl}\n")
        f.write(f"  - Augmentation: {config.use_augmentation} (x{config.augmentation_multiplier})\n")
        f.write(f"  - Synthetic Data: {config.generate_synthetic} ({config.synthetic_samples})\n")
        f.write(f"  - LoRA rank: {config.lora_r}\n")
        f.write(f"  - Learning rate: {config.learning_rate}\n")
        f.write(f"  - Batch size: {config.batch_size * config.gradient_accumulation_steps}\n")
    
    print(f"Training summary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_scl", action="store_true", help="Disable SCL")
    parser.add_argument("--no_aug", action="store_true", help="Disable data augmentation")
    parser.add_argument("--no_synthetic", action="store_true", help="Disable synthetic data")
    parser.add_argument("--num_epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    
    args = parser.parse_args()
    
    config = Config()
    
    if args.no_scl:
        config.use_scl = False
    if args.no_aug:
        config.use_augmentation = False
    if args.no_synthetic:
        config.generate_synthetic = False
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.output_dir:
        config.output_dir = args.output_dir
    
    main(config)