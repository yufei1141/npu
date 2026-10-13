---
license: apache-2.0
base_model: Qwen/Qwen3-8B
quantized_by: 6block
pipeline_tag: text-generation
tags:
  - compressed-tensors
  - GPTQ
  - vllm
  - quantized
language:
  - en
  - zh
---

# Qwen3-8B-GPTQ

GPTQ (W4A16) quantization of [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), produced
with [llm-compressor](https://github.com/vllm-project/llm-compressor) in the **compressed-tensors**
format for efficient serving with vLLM and SGLang.

> 4-bit error-compensated weight quantization; roughly a quarter of the bf16 size.

## Serving / 部署

```bash
vllm serve 6block/Qwen3-8B-GPTQ
```

The quantization scheme is declared in `config.json` and detected automatically — no extra flags are
required. Compatible with any engine that reads compressed-tensors (vLLM, SGLang, …).

`config.json` 中已声明量化方案，引擎自动识别，无需额外参数。兼容一切支持 compressed-tensors 的推理框架。

## Prompt format / 对话格式

```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
```

## Details / 说明

| | |
|---|---|
| Method / 方法 | GPTQ (W4A16) |
| Format / 格式 | compressed-tensors (safetensors) |
| Base model / 原始模型 | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| Tool / 工具 | [llm-compressor](https://github.com/vllm-project/llm-compressor) |
| License / 许可 | apache-2.0 (inherited from base model / 继承自原始模型) |

**English** — This checkpoint reproduces the base model's behavior at reduced precision, shrinking
memory footprint and increasing throughput while keeping quality close to the original. It is a
drop-in replacement for the base model in vLLM/SGLang deployments.

**中文** — 本权重是原始模型的低精度量化版本，在保持接近原模型质量的同时，显著降低显存占用、提升吞吐。
在 vLLM/SGLang 部署中可直接替换原始模型使用。

---

*Quantized by 6block. Weights are unmodified apart from quantization; all model behavior and license
obligations follow the base model. / 由 6block 量化，除量化外未改动权重，模型行为与许可义务均遵循原始模型。*
