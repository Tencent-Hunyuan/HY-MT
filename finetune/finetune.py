# Copyright 2024 Tencent Inc. All Rights Reserved.
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

# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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


import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import torch
import shutil
import logging
from dataclasses import dataclass, field
import deepspeed
from typing import Optional, Dict

import transformers
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model, PeftModel
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.modeling_utils import unwrap_model
from typing import Literal


def print_args(args, name='arguments'):
    """Print arguments."""
    if torch.distributed.get_rank() == 0:
        print(f'------------------------ {name} ------------------------', flush=True)
        str_list = []
        for arg in vars(args):
            dots = '.' * (48 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print(f'-------------------- end of {name} ---------------------', flush=True)


@dataclass
class ModelArguments:
    use_flash_attn: bool = field(
        default=False, 
        metadata={"help": "Enable FlashAttention-2 for faster training."}
    )
    use_lora: bool = field(default=False, metadata={"help": "Enable Lora for faster training."})
    hidden_size: int = field(default=2048, metadata={"help": "The hidden size of the model."})
    num_layers: int = field(default=24, metadata={"help": "The number of layers of the model."})
    num_attention_heads: int = field(default=16, metadata={"help": "The number of attention heads of the model."})
    intermediate_size: int = field(default=8192, metadata={"help": "The intermediate size of the model."})
    max_position_embeddings: int = field(
        default=2048, 
        metadata={"help": "The maximum sequence length that this model might ever be used with."}
    )
    vocab_size: int = field(default=50257, metadata={"help": "The vocabulary size of the model."})
    type_vocab_size: int = field(default=1, metadata={"help": "The vocabulary size of the model."})
    layer_norm_eps: float = field(
        default=1e-5, 
        metadata={"help": "The epsilon used by the layer normalization layers of the model."}
    )
    moe_topk: int = field(default=4, metadata={"help": "The topk for MOE."})
    num_experts: int = field(default=8, metadata={"help": "The number of experts for MOE."})
    num_key_value_heads: int = field(default=16, metadata={"help": "The number of key-value heads in GQA."})
    use_cla: bool = field(default=False, metadata={"help": "Whether to use CLA."})
    cla_share_factor: int = field(default=2, metadata={"help": "The share factor for CLA."})
    use_mixed_mlp_moe: bool = field(
        default=False, 
        metadata={"help": "Whether to use mixed MoE with shared expert."}
    )
    num_shared_expert: int = field(default=1, metadata={"help": "Number of shared experts."})
    use_qk_norm: bool = field(default=False, metadata={"help": "Whether to use qk norm."})
    tie_word_embeddings: bool = field(
        default=True, 
        metadata={"help": "Whether to tie the word embeddings of the encoder and the decoder."}
    )
    lora_rank: int = field(default=64, metadata={"help": "The rank of lora."})
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    train_attention_params_only: bool = field(default=False, metadata={
        "help": "Whether to train attention parameters only."}
    )


@dataclass
class DataArguments:
    model_size : Literal["0.5B", "1.8B", "4B", "7B"] = field(
        default="7B",
        metadata={"help": "Select the model size from ['0.5B', '1.8B', '4B', '7B']"}
    )
    train_data_file: str = field(default=None, metadata={"help": "Path to the training data."})
    max_seq_length: int = field(
        default=2048, 
        metadata={"help": "The max sequence length of the model inputs after tokenization."}
    )
    complex_data: Optional[str] = field(default=None)
    use_dummy_data: bool = field(default=False, metadata={"help": "Use dummy data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    tokenizer_name_or_path: Optional[str] = field(default=None)
    model_name_or_path: Optional[str] = field(default=None)
    make_moe_param_leaf_module: bool = field(
        default=False, 
        metadata={"help": "Make MoE parameters zero-3 leaf module."}
    )
    min_lr: float = field(
        default=0.01, 
        metadata={"help": "The final learning rate at the end of the decay will be learning_rate * min_lr"}
    ),
    average_tokens_across_devices : bool = field(default=False)


IGNORE_INDEX = -100


class DummyDataset(Dataset):
    def __init__(self, tokenizer, max_seq_length=512, length=1000):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.length = length
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        tokens = torch.randint(0, self.tokenizer.vocab_size, (self.max_seq_length, ))
        return {'input_ids': tokens, 'labels': tokens}


class SFTDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_seq_length=2048, model_size="7B", prompt_format='mplus'):
        self.tokenizer = tokenizer
        self.prompt_format = prompt_format
        self.max_seq_length = max_seq_length
        self.data_list = self.load_data(data_file)
        self.model_size = model_size

    def __len__(self):
        return len(self.data_list)

    def load_data(self, data_file):
        logging.info('Loading data: {}'.format(data_file))
        with open(data_file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        logging.info("there are {} data in dataset".format(len(data_list)))
        return data_list

    def encode_data(self, data_dict):
        model_inputs = {}
        message_tokens = torch.tensor(self.tokenizer.apply_chat_template(data_dict['messages']))

        # Note: The 7B model uses a different vocabulary than other models.
        if self.model_size in "7B":
            sep_token_id = self.tokenizer.convert_tokens_to_ids('<|extra_0|>')
            eos_token_id = self.tokenizer.convert_tokens_to_ids('<|eos|>')
        else:
            sep_token_id = self.tokenizer.convert_tokens_to_ids('<｜hy_Assistant｜>')
            eos_token_id = self.tokenizer.convert_tokens_to_ids('<｜hy_place▁holder▁no▁2｜>')
        
        loss_token_begins = (message_tokens == sep_token_id).nonzero(as_tuple=True)[0].tolist()
        loss_token_ends = (message_tokens == eos_token_id).nonzero(as_tuple=True)[0].tolist()
        message_labels = torch.tensor([IGNORE_INDEX] * message_tokens.shape[0])
        for begin_idx, end_idx in zip(loss_token_begins, loss_token_ends):
            message_labels[begin_idx:end_idx + 1] = message_tokens[begin_idx:end_idx + 1]
        input_ids = message_tokens.to(torch.long)
        labels = message_labels.to(torch.long)

        input_ids = input_ids[:self.max_seq_length]
        labels = labels[:self.max_seq_length]
        attention_mask = [1 if val != self.tokenizer.encode(self.tokenizer.pad_token)[0] else 0 for val in input_ids]
        model_inputs["input_ids"] = input_ids
        model_inputs["attention_mask"] = torch.tensor(attention_mask, dtype=torch.bool)
        model_inputs["labels"] = labels

        return model_inputs

    def __getitem__(self, index):
        data = self.data_list[index]
        data = json.loads(data)
        model_inputs = self.encode_data(data)

        return model_inputs


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids = [instance['input_ids'] for instance in instances]
        labels = [instance['labels'] for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.encode(self.tokenizer.pad_token)[0])
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.encode(self.tokenizer.pad_token)[0]),
        )


def make_supervised_data_module(tokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if data_args.use_dummy_data:
        train_dataset = DummyDataset(tokenizer, data_args.max_seq_length)
    else:
        train_dataset = SFTDataset(
            tokenizer=tokenizer, 
            data_file=data_args.train_data_file, 
            max_seq_length=data_args.max_seq_length,
            model_size=data_args.model_size
        )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print_args(model_args, 'model arguments')
    print_args(data_args, 'data arguments')
    print_args(training_args, 'training arguments')

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        training_args.tokenizer_name_or_path,
        trust_remote_code=True
    )

    init_kwargs = {}
    if model_args.use_flash_attn:
        init_kwargs["attn_implementation"] = "flash_attention_2"
    if training_args.bf16:
        init_kwargs["torch_dtype"] = torch.bfloat16
    elif training_args.fp16:
        init_kwargs["torch_dtype"] = torch.float16

    if training_args.model_name_or_path is not None and os.path.exists(training_args.model_name_or_path):
        print(f"Initializing model from local file: {training_args.model_name_or_path}")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            training_args.model_name_or_path,
            trust_remote_code=True,
            **init_kwargs
        )
    else:
        raise FileNotFoundError(f"Model path {training_args.model_name_or_path} is invalid or does not exist.")
    
    if model_args.train_attention_params_only:
        for name, param in model.named_parameters():
            if 'self_attn' not in name:
                param.requires_grad = False

    if model_args.use_lora:
        # define Lora configuration
        lora_config = LoraConfig(
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    
    # donnot split MoE weights when using zero3
    if model_args.num_experts > 0 \
        and training_args.make_moe_param_leaf_module and \
            training_args.deepspeed_plugin.zero_stage == 3:
        from deepspeed.utils import set_z3_leaf_modules
        set_z3_leaf_modules(model, [HunYuanMoE])

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # Tell Trainer not to attempt DataParallel
    model.is_parallelizable = True
    model.model_parallel = True

    training_args.lr_scheduler_kwargs = {
        'min_lr': training_args.min_lr,
    }

    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args,
        **data_module
    )
    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)


if __name__ == "__main__":
    train()