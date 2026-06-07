"""Reference BLT loading utilities for Korean GEC fine-tuning."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class ReferenceBltComponents:
    model: torch.nn.Module
    tokenizer: object
    patcher: object
    entropy_model: torch.nn.Module


class ReferenceBltUnavailable(RuntimeError):
    """Raised when the reference BLT dependency chain is not available."""


def add_reference_blt_to_path(reference_code_dir: str | Path) -> None:
    ref_dir = Path(reference_code_dir)
    if not ref_dir.exists():
        raise ReferenceBltUnavailable(f"Reference BLT directory not found: {ref_dir}")
    ref_dir_str = str(ref_dir.resolve())
    if ref_dir_str not in sys.path:
        sys.path.insert(0, ref_dir_str)


def _bool_local_files_only(value: bool) -> bool:
    return bool(value)


def load_reference_blt_components(
    *,
    reference_code_dir: str | Path = "reference_code/blt",
    blt_repo: str = "facebook/blt-1b",
    entropy_repo: str = "facebook/blt-entropy",
    local_files_only: bool = False,
    device: torch.device | str = "cuda",
    precision: str = "bf16",
) -> ReferenceBltComponents:
    """Load official BLT model, tokenizer, entropy model, and dynamic patcher.

    This intentionally has no mini fallback. If reference BLT cannot be loaded,
    training must fail instead of silently running a byte-only model.
    """

    add_reference_blt_to_path(reference_code_dir)
    try:
        from bytelatent.data.patcher import to_device
        from bytelatent.hf import BltTokenizerAndPatcher
        from bytelatent.model.blt import ByteLatentTransformer
        from bytelatent.transformer import LMTransformer
    except Exception as exc:  # pragma: no cover - depends on cluster BLT env
        raise ReferenceBltUnavailable(
            "Could not import reference BLT. Install reference_code/blt requirements "
            "including xformers, or use the BLT conda environment."
        ) from exc

    local_only = _bool_local_files_only(local_files_only)
    try:
        model = ByteLatentTransformer.from_pretrained(
            blt_repo,
            local_files_only=local_only,
        )
        entropy_model = LMTransformer.from_pretrained(
            entropy_repo,
            local_files_only=local_only,
        )
        tok_and_patcher = BltTokenizerAndPatcher.from_pretrained(
            blt_repo,
            local_files_only=local_only,
        )
    except Exception as exc:  # pragma: no cover - depends on HF auth/cache
        raise ReferenceBltUnavailable(
            f"Could not load BLT weights/tokenizer from blt_repo={blt_repo!r}, "
            f"entropy_repo={entropy_repo!r}. Ensure HF access is approved and "
            "weights are cached or available online."
        ) from exc

    tokenizer = tok_and_patcher.tokenizer_args.build()
    patcher_args = tok_and_patcher.patcher_args.model_copy(deep=True)
    patcher_args.realtime_patching = False
    patch_device = "cuda" if torch.device(device).type == "cuda" else "cpu"
    patcher_args.patching_device = patch_device
    patcher_args.device = patch_device
    patcher = patcher_args.build()

    dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[precision]
    model = model.to(device)
    entropy_model = entropy_model.eval()
    for param in model.parameters():
        param.data = param.data.to(dtype=dtype)
    for param in entropy_model.parameters():
        param.requires_grad = False

    patcher.realtime_patching = True
    patcher.entropy_model, _ = to_device(entropy_model, patcher_args.patching_device)
    if patcher.patching_mode != "entropy":
        raise ReferenceBltUnavailable(
            f"Expected entropy dynamic patching, got patching_mode={patcher.patching_mode!r}"
        )

    _override_attn_impl(model, "sdpa")
    if hasattr(model, "cross_attn_use_flex_attention"):
        model.cross_attn_use_flex_attention = False
    _patch_cross_attention_to_sdpa(model)

    return ReferenceBltComponents(
        model=model,
        tokenizer=tokenizer,
        patcher=patcher,
        entropy_model=entropy_model,
    )


def _override_attn_impl(module: torch.nn.Module, impl: str = "sdpa") -> None:
    if hasattr(module, "attn_impl"):
        module.attn_impl = impl
    for child in module.children():
        _override_attn_impl(child, impl)


def _cross_attention_forward_sdpa(self, x, kv, mask=None):
    """sdpa replacement for CrossAttention.forward (avoids flex_attention)."""
    bsz, seq_len, _ = x.shape
    _, slen_kv, _ = kv.shape
    x_norm = self.cross_attn_norm_q(x)
    kv = self.cross_attn_norm_kv(kv)

    xq = self.wq(x_norm)
    xk = self.wk(kv)
    xv = self.wv(kv)

    output_shape = xq.shape
    xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
    xk = xk.view(bsz, slen_kv, self.n_kv_heads, self.head_dim)
    xv = xv.view(bsz, slen_kv, self.n_kv_heads, self.head_dim)

    from bytelatent.base_transformer import repeat_kv
    xk = repeat_kv(xk, self.heads_per_group, dim=2)
    xv = repeat_kv(xv, self.heads_per_group, dim=2)

    xq, xk, xv = (e.transpose(1, 2) for e in (xq, xk, xv))
    attn_mask = mask if isinstance(mask, torch.Tensor) else None
    output = torch.nn.functional.scaled_dot_product_attention(
        xq, xk, xv, attn_mask=attn_mask,
    )
    output = output.transpose(1, 2).contiguous()
    output = self.wo(output.reshape(output_shape))
    return x + output


def _patch_cross_attention_to_sdpa(module: torch.nn.Module) -> None:
    import types
    try:
        from bytelatent.model.latent_transformer import CrossAttention
    except ImportError:
        return
    for child in module.modules():
        if isinstance(child, CrossAttention):
            child.forward = types.MethodType(_cross_attention_forward_sdpa, child)
