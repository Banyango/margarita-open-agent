from enum import StrEnum


class LLMModelEnum(StrEnum):
    # Gemma
    GEMMA_4_LATEST = "gemma4:latest"
    GEMMA_4_E2B = "gemma4:e2b"
    GEMMA_4_E4B = "gemma4:e4b"
    GEMMA_4_26B = "gemma4:26b"
    GEMMA_4_31B = "gemma4:31b"
    GEMMA_4_31B_CLOUD = "gemma4:31b-cloud"

    # DeepSeek
    DEEPSEEK_R1_7B = "deepseek-r1:7b"
    DEEPSEEK_V3_1_671B = "deepseek-v3.1:671b"
    DEEPSEEK_V3_2_CLOUD = "deepseek-v3.2:cloud"
    DEEPSEEK_V4_FLASH_CLOUD = "deepseek-v4-flash:cloud"

    # Qwen3
    QWEN3_8B = "qwen3:8b"
    QWEN3_5_LATEST = "qwen3.5:latest"
    QWEN3_5_0_8B = "qwen3.5:0.8b"
    QWEN3_5_2B = "qwen3.5:2b"
    QWEN3_5_4B = "qwen3.5:4b"
    QWEN3_5_9B = "qwen3.5:9b"
    QWEN3_5_27B = "qwen3.5:27b"
    QWEN3_5_35B = "qwen3.5:35b"
    QWEN3_5_122B = "qwen3.5:122b"
    QWEN3_5_CLOUD = "qwen3.5:cloud"
    QWEN3_5_397B_CLOUD = "qwen3.5:397b-cloud"
    QWEN3_6_LATEST = "qwen3.6:latest"
    QWEN3_6_27B = "qwen3.6:27b"
    QWEN3_6_35B = "qwen3.6:35b"
    QWEN3_6_27B_CODING_MXFP8 = "qwen3.6:27b-coding-mxfp8"
    QWEN3_6_27B_CODING_NVFP4 = "qwen3.6:27b-coding-nvfp4"
    QWEN3_6_27B_CODING_BF16 = "qwen3.6:27b-coding-bf16"
    QWEN3_6_27B_MLX_BF16 = "qwen3.6:27b-mlx-bf16"
    QWEN3_6_35B_A3B_CODING_MXFP8 = "qwen3.6:35b-a3b-coding-mxfp8"
    QWEN3_6_35B_A3B_CODING_NVFP4 = "qwen3.6:35b-a3b-coding-nvfp4"
    QWEN3_6_35B_A3B_CODING_BF16 = "qwen3.6:35b-a3b-coding-bf16"
    QWEN3_6_35B_A3B_MLX_BF16 = "qwen3.6:35b-a3b-mlx-bf16"
    QWEN3_NEXT_LATEST = "qwen3-next:latest"
    QWEN3_NEXT_80B = "qwen3-next:80b"
    QWEN3_NEXT_80B_CLOUD = "qwen3-next:80b-cloud"
    QWEN3_VL_8B = "qwen3-vl:8b"

    # GLM
    GLM_4_6_CLOUD = "glm-4.6:cloud"
    GLM_4_7_CLOUD = "glm-4.7:cloud"
    GLM_4_7_FLASH = "glm-4.7-flash:latest"
    GLM_4_7_FLASH_Q4_K_M = "glm-4.7-flash:q4_K_M"
    GLM_4_7_FLASH_Q8_0 = "glm-4.7-flash:q8_0"
    GLM_4_7_FLASH_BF16 = "glm-4.7-flash:bf16"
    GLM_5_CLOUD = "glm-5:cloud"
    GLM_5_1_CLOUD = "glm-5.1:cloud"

    # Kimi
    KIMI_K2_THINKING_CLOUD = "kimi-k2-thinking:cloud"
    KIMI_K2_5_CLOUD = "kimi-k2.5:cloud"
    KIMI_K2_6_CLOUD = "kimi-k2.6:cloud"

    # MiniMax
    MINIMAX_M2_CLOUD = "minimax-m2:cloud"
    MINIMAX_M2_5_CLOUD = "minimax-m2.5:cloud"
    MINIMAX_M2_7_CLOUD = "minimax-m2.7:cloud"

    # Nemotron
    NEMOTRON_3_NANO_LATEST = "nemotron-3-nano:latest"
    NEMOTRON_3_NANO_4B = "nemotron-3-nano:4b"
    NEMOTRON_3_NANO_30B = "nemotron-3-nano:30b"
    NEMOTRON_3_NANO_30B_CLOUD = "nemotron-3-nano:30b-cloud"
    NEMOTRON_3_SUPER_LATEST = "nemotron-3-super:latest"
    NEMOTRON_3_SUPER_120B = "nemotron-3-super:120b"
    NEMOTRON_3_SUPER_CLOUD = "nemotron-3-super:cloud"
    NEMOTRON_CASCADE_2_LATEST = "nemotron-cascade-2:latest"
    NEMOTRON_CASCADE_2_30B = "nemotron-cascade-2:30b"

    # Mistral
    MAGISTRAL_24B = "magistral:24b"

    # Google
    GEMINI_3_FLASH_PREVIEW_LATEST = "gemini-3-flash-preview:latest"
    GEMINI_3_FLASH_PREVIEW_CLOUD = "gemini-3-flash-preview:cloud"

    # OpenAI
    GPT_OSS_20B = "gpt-oss:20b"
    GPT_OSS_SAFEGUARD_LATEST = "gpt-oss-safeguard:latest"
    GPT_OSS_SAFEGUARD_20B = "gpt-oss-safeguard:20b"
    GPT_OSS_SAFEGUARD_120B = "gpt-oss-safeguard:120b"
