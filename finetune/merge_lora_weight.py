# 导入所需的库
from transformers import AutoModelForCausalLM  # 用于加载预训练的语言模型
from peft import LoraConfig, get_peft_model, PeftModel  # 用于处理LoRA权重
import argparse  # 用于解析命令行参数
import shutil  # 用于文件操作，如复制
import os  # 用于文件路径操作
import torch  # 用于深度学习操作

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument("--base_model_path", type=str, required=True, 
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--adapter_model_path", type=str, required=True, help="Path to adapter model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output model")
    parser.add_argument("--save_dtype", type=str, choices=['bf16', 'fp32', 'fp16'], 
                        default='fp32', help="In which dtype to save, fp32, bf16 or fp16.")
    # 解析命令行参数
    args = parser.parse_args()

    name2dtype = {'bf16': torch.bfloat16, 'fp32': torch.float32, 'fp16': torch.float16}
    # 加载基座模型
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, device_map='cpu', 
        trust_remote_code=True, torch_dtype=name2dtype[args.save_dtype]
    )
    # 在基座模型的基础上加载 adapter 权重
    model = PeftModel.from_pretrained(model, args.adapter_model_path, trust_remote_code=True)
    # 融合模型和 adapter
    model = model.merge_and_unload()
    # 保存融合后的模型权重
    model.save_pretrained(args.output_path, safe_serialization=False)

    # 拷贝 tokenizer，config 和模型文件
    shutil.copy(
        os.path.join(args.base_model_path, 'generation_config.json'), 
        os.path.join(args.output_path, 'generation_config.json')
    )
    shutil.copy(
        os.path.join(args.base_model_path, 'hy.tiktoken'), 
        os.path.join(args.output_path, 'hy.tiktoken')
    )
    shutil.copy(
        os.path.join(args.base_model_path, 'tokenizer_config.json'), 
        os.path.join(args.output_path, 'tokenizer_config.json')
    )
    shutil.copy(
        os.path.join(args.base_model_path, 'config.json'), 
        os.path.join(args.output_path, 'config.json')
    )
    shutil.copy(
        os.path.join(args.base_model_path, 'modeling_hunyuan.py'), 
        os.path.join(args.output_path, 'modeling_hunyuan.py')
    )
    shutil.copy(
        os.path.join(args.base_model_path, 'configuration_hunyuan.py'), 
        os.path.join(args.output_path, 'configuration_hunyuan.py')
    )
    shutil.copy(
        os.path.join(args.base_model_path, 'tokenization_hy.py'), 
        os.path.join(args.output_path, 'tokenization_hy.py')
    )

    print(f'Merged model weight is saved to {args.output_path}')
    
if __name__ == "__main__":
    main()
