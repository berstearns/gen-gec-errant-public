"""Generation runner: load models, generate continuations, compute perplexity."""

import logging
import math
from typing import List

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

from gen_gec_errant.generation.config import GenerationConfig, GenerationParams, ModelConfig

logger = logging.getLogger(__name__)


def get_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def load_model(model_config: ModelConfig, device: torch.device) -> tuple:
    """Load a model and tokenizer, applying learner checkpoint if provided."""
    logger.info("Loading model: %s (%s)", model_config.name, model_config.hf_model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_config.hf_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_config.hf_model_id, torch_dtype=dtype,
    )

    if model_config.is_learner_tuned and model_config.checkpoint_path:
        logger.info("Loading learner checkpoint from: %s", model_config.checkpoint_path)
        checkpoint = torch.load(model_config.checkpoint_path, map_location="cpu")

        if isinstance(checkpoint, dict):
            state_dict = (
                checkpoint.get("model_state_dict")
                or checkpoint.get("state_dict")
                or checkpoint.get("model")
                or checkpoint
            )
        else:
            state_dict = checkpoint

        # Strip _orig_mod. prefix (torch.compile artifact)
        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k.replace("_orig_mod.", "")] = v
        if any(k.startswith("_orig_mod.") for k in state_dict):
            logger.info("Stripped '_orig_mod.' prefix from checkpoint keys")
        state_dict = cleaned

        # Transpose weights for nanoGPT -> HF Conv1D compatibility
        model_state = model.state_dict()
        for k, v in state_dict.items():
            if k in model_state and v.shape != model_state[k].shape:
                if v.dim() == 2 and v.shape == model_state[k].shape[::-1]:
                    state_dict[k] = v.t()

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing keys in checkpoint: %s...", missing[:5])
        if unexpected:
            logger.warning("Unexpected keys in checkpoint: %s...", unexpected[:5])
        logger.info("Learner checkpoint loaded successfully")

    model = model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_continuations(
    model,
    tokenizer,
    prompts: List[str],
    gen_params: GenerationParams,
    batch_size: int = 8,
    device: torch.device = torch.device("cpu"),
) -> List[str]:
    """Generate text continuations for a list of prompts."""
    continuations = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        prompt_lengths = inputs["attention_mask"].sum(dim=1)

        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_params.max_new_tokens,
            min_new_tokens=gen_params.min_new_tokens,
            temperature=gen_params.temperature,
            top_k=gen_params.top_k,
            top_p=gen_params.top_p,
            do_sample=gen_params.do_sample,
            num_return_sequences=gen_params.num_return_sequences,
            repetition_penalty=gen_params.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
        )

        for j, output_ids in enumerate(outputs):
            prompt_len = prompt_lengths[j].item()
            continuation_ids = output_ids[prompt_len:]
            continuation = tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()
            continuations.append(continuation)

        if (i // batch_size) % 10 == 0:
            logger.info("  Generated %d/%d", min(i + batch_size, len(prompts)), len(prompts))

    return continuations


@torch.no_grad()
def compute_perplexity(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int = 8,
    device: torch.device = torch.device("cpu"),
) -> List[float]:
    """Compute per-sentence perplexity for a list of texts."""
    perplexities = []
    loss_fn = CrossEntropyLoss(reduction="none")

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        outputs = model(**inputs, labels=inputs["input_ids"])

        logits = outputs.logits[:, :-1, :].contiguous()
        labels = inputs["input_ids"][:, 1:].contiguous()
        attention_mask = inputs["attention_mask"][:, 1:].contiguous()

        batch_sz, seq_len, vocab_size = logits.shape
        flat_logits = logits.view(-1, vocab_size)
        flat_labels = labels.view(-1)

        token_losses = loss_fn(flat_logits, flat_labels).view(batch_sz, seq_len)
        token_losses = token_losses * attention_mask.float()
        seq_lengths = attention_mask.sum(dim=1).float()
        mean_losses = token_losses.sum(dim=1) / seq_lengths.clamp(min=1)

        for loss_val in mean_losses:
            perplexities.append(math.exp(loss_val.item()))

    return perplexities


def run_generation(config: GenerationConfig, items: List[dict]) -> dict:
    """
    Run the generation stage for a single model.

    Args:
        config: GenerationConfig with model and generation parameters
        items: List of dicts from data_loader (prompt, reference, full)

    Returns:
        Dict with keys: continuations, full_texts, perplexities, model_name
    """
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = get_device(config.device)
    logger.info("Using device: %s", device)

    model, tokenizer = load_model(config.model, device)

    prompts = [item["prompt"] for item in items]

    logger.info("Generating with %s...", config.model.name)
    continuations = generate_continuations(
        model, tokenizer, prompts, config.params,
        batch_size=config.batch_size, device=device,
    )

    full_texts = [f"{p} {c}" for p, c in zip(prompts, continuations)]

    logger.info("Computing perplexity for %s...", config.model.name)
    perplexities = compute_perplexity(
        model, tokenizer, full_texts,
        batch_size=config.batch_size, device=device,
    )

    logger.info("Mean perplexity: %.2f", sum(perplexities) / len(perplexities))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "model_name": config.model.name,
        "continuations": continuations,
        "full_texts": full_texts,
        "perplexities": perplexities,
    }
