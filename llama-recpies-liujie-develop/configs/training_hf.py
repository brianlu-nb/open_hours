# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class train_config_hf:
    model_name: str="PATH/to/LLAMA/7B"
    enable_fsdp: bool=False
    low_cpu_fsdp: bool=False
    run_validation: bool=True
    batch_size_training: int=4
    num_epochs: int=3
    num_workers_dataloader: int=1
    lr: float=1e-4
    weight_decay: float=0.0
    gamma: float= 0.85
    seed: int=42
    use_bf16: bool=False
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size: int=1
    dataset = "samsum_dataset"
    micro_batch_size: int=4
    peft_method: str = "lora" # None , llama_adapter, prefix
    use_peft: bool=False
    output_dir: str = "PATH/to/save/PEFT/model"
    quantization: bool = False
    save_optimizer: bool=False # will be used if using FSDP
    use_fast_kernels: bool = True, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    save_steps: int = 500 # save checkpoint every x steps
    optimizer: str = "adamw_torch"
    max_length: int = None
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    
    