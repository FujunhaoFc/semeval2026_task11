"""
SemEval 2026 Task 11 - DeBERTa + SCL + LoRA Baseline
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
import argparse
from pathlib import Path
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoTokenizer,
    DebertaV2Model,
    AdamW,
    get_cosine_schedule_with_warmup
)
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import accuracy_score


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Training configuration"""
    # Model
    model_name = "microsoft/deberta-v3-large"
    max_length = 256
    use_scl = True
    scl_temperature = 0.07
    scl_projection_dim = 256
    
    # LoRA
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.1
    
    # Training
    num_epochs = 10
    batch_size = 8
    learning_rate = 3e-4
    warmup_ratio = 0.1
    weight_decay = 0.01
    scl_loss_weight = 0.5 
    
    # Data
    data_path = "pilot_data/syllogistic_reasoning_binary_pilot_en.json"
    train_split = 0.9
    
    # Output
    output_dir = "outputs"
    save_steps = 100
    
    # Misc
    seed = 42


# ============================================================================
# Dataset
# ============================================================================

class SyllogismDataset(Dataset):
    """Simple dataset for syllogistic reasoning"""
    
    def __init__(self, data_path, tokenizer, max_length=256, augment=False):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        
        # Group by validity for contrastive learning
        self.valid_indices = [i for i, item in enumerate(self.data) if item['validity']]
        self.invalid_indices = [i for i, item in enumerate(self.data) if not item['validity']]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['syllogism']
        label = 1 if item['validity'] else 0
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
        }
        
        # Get positive and negative samples for SCL
        if self.augment:
            # Positive: same validity
            pos_pool = self.valid_indices if label == 1 else self.invalid_indices
            pos_idx = random.choice([i for i in pos_pool if i != idx])
            pos_text = self.data[pos_idx]['syllogism']
            
            # Negative: different validity
            neg_pool = self.invalid_indices if label == 1 else self.valid_indices
            neg_idx = random.choice(neg_pool)
            neg_text = self.data[neg_idx]['syllogism']
            
            # Tokenize positive
            pos_enc = self.tokenizer(
                pos_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Tokenize negative
            neg_enc = self.tokenizer(
                neg_text,
                max_length=self.max_length,
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
            task_type=TaskType.SEQ_CLS,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["query_proj", "value_proj", "key_proj", "dense"],
            bias="none",
        )
        self.deberta = get_peft_model(self.deberta, peft_config)
        
        # Classification head
        self.classifier = nn.Linear(hidden_size, 2)
        
        # Projection head for SCL
        if config.use_scl:
            self.projection_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, config.scl_projection_dim)
            )
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, attention_mask, labels=None,
                positive_input_ids=None, positive_attention_mask=None,
                negative_input_ids=None, negative_attention_mask=None):
        
        # Encode anchor
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        pooled = self.dropout(pooled)
        
        # Classification
        logits = self.classifier(pooled)
        
        loss = None
        if labels is not None:
            # Classification loss
            ce_loss = F.cross_entropy(logits, labels)
            loss = ce_loss
            
            # Contrastive loss
            if self.config.use_scl and positive_input_ids is not None:
                # Encode positive
                pos_outputs = self.deberta(input_ids=positive_input_ids, 
                                          attention_mask=positive_attention_mask)
                pos_pooled = pos_outputs.last_hidden_state[:, 0, :]
                
                # Encode negative
                neg_outputs = self.deberta(input_ids=negative_input_ids,
                                          attention_mask=negative_attention_mask)
                neg_pooled = neg_outputs.last_hidden_state[:, 0, :]
                
                # Project to contrastive space
                anchor_proj = F.normalize(self.projection_head(pooled), dim=1)
                pos_proj = F.normalize(self.projection_head(pos_pooled), dim=1)
                neg_proj = F.normalize(self.projection_head(neg_pooled), dim=1)
                
                # Compute similarities
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
# Training
# ============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    predictions = []
    labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        batch_labels = batch['labels'].to(device)
        
        # Get positive/negative if available
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
        
        loss = outputs['loss']
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Metrics
        total_loss += loss.item()
        preds = torch.argmax(outputs['logits'], dim=1)
        predictions.extend(preds.cpu().numpy())
        labels.extend(batch_labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
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
    
    return avg_loss, accuracy, predictions, labels


def main(config):
    # Setup
    set_seed(config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(config.output_dir, exist_ok=True)
    
    print("="*50)
    print("SemEval 2026 Task 11 - Baseline Training")
    print("="*50)
    print(f"Device: {device}")
    print(f"Model: {config.model_name}")
    print(f"Use SCL: {config.use_scl}")
    print(f"Data: {config.data_path}")
    print("="*50)
    
    # Load tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # Load data
    print("[2/5] Loading data...")
    full_dataset = SyllogismDataset(
        config.data_path, 
        tokenizer, 
        config.max_length,
        augment=config.use_scl
    )
    
    train_size = int(config.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # Create model
    print("[3/5] Creating model...")
    model = DeBertaSCL(config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total: {total_params:,} | Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Optimizer and scheduler
    print("[4/5] Setting up optimizer...")
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    num_training_steps = len(train_loader) * config.num_epochs
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    print("[5/5] Training...")
    print("="*50)
    
    best_val_acc = 0
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device
        )
        
        # Evaluate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device)
        
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = Path(config.output_dir) / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, save_path)
            print(f"  ✓ Saved best model (acc: {val_acc:.4f})")
    
    print("\n" + "="*50)
    print(f"Training completed! Best Val Acc: {best_val_acc:.4f}")
    print(f"Model saved to: {config.output_dir}/best_model.pt")
    print("="*50)
    
    # Save predictions for official evaluation
    print("\nGenerating predictions for evaluation...")
    model.load_state_dict(torch.load(Path(config.output_dir) / "best_model.pt")['model_state_dict'])
    _, _, predictions, true_labels = evaluate(model, val_loader, device)
    
    # Save in format for official evaluation script
    eval_data = []
    for i, (pred, true) in enumerate(zip(predictions, true_labels)):
        eval_data.append({
            "id": str(i),
            "prediction": bool(pred)  # Convert to boolean
        })
    
    pred_path = Path(config.output_dir) / "predictions.json"
    with open(pred_path, 'w') as f:
        json.dump(eval_data, f, indent=2)
    
    print(f"Predictions saved to: {pred_path}")
    print("\nUse official evaluation script:")
    print(f"  python evaluation_kit/task_1_3/evaluation_script.py \\")
    print(f"    --gold_file {config.data_path} \\")
    print(f"    --pred_file {pred_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--no_scl", action="store_true", help="Disable SCL")
    
    args = parser.parse_args()
    
    # Create config
    config = Config()
    
    # Override with command line args
    if args.data_path:
        config.data_path = args.data_path
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.no_scl:
        config.use_scl = False
    
    main(config)