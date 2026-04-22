"""
한국어 GEC 추론 엔진.
HuggingFace 공개 모델 Soyoung97/gec_kr (KoBART fine-tuned) 사용.
"""
import torch
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast


class GecInferenceEngine:
    """KoBART GEC 모델 로드 및 추론."""

    def __init__(
        self,
        model_name: str = "Soyoung97/gec_kr",
        device: str | None = None,
        num_beams: int = 4,
        max_length: int = 128,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_beams = num_beams
        self.max_length = max_length

        print(f"Loading model '{model_name}' on {self.device} ...")
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device).eval()
        print("Model loaded.")

    @torch.inference_mode()
    def correct(self, text: str) -> str:
        """단일 문장 교정."""
        if not text.strip():
            return ""

        raw_ids = self.tokenizer.encode(text)
        input_ids = [self.tokenizer.bos_token_id] + raw_ids + [self.tokenizer.eos_token_id]
        input_tensor = torch.tensor([input_ids], device=self.device)

        output_ids = self.model.generate(
            input_tensor,
            max_length=self.max_length,
            num_beams=self.num_beams,
            eos_token_id=1,
            early_stopping=True,
            repetition_penalty=2.0,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    @torch.inference_mode()
    def correct_batch(self, texts: list[str]) -> list[str]:
        """여러 문장 배치 교정."""
        results = []
        for text in texts:
            results.append(self.correct(text))
        return results


def make_diff(original: str, corrected: str) -> list[tuple[str, str | None]]:
    """원문과 교정문의 문자 단위 diff를 생성.
    반환: [(text, tag), ...] 형태 — tag는 None(동일), '+', '-'
    """
    import difflib

    diff = []
    matcher = difflib.SequenceMatcher(None, original, corrected)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            diff.append((original[i1:i2], None))
        elif op == "replace":
            diff.append((original[i1:i2], "-"))
            diff.append((corrected[j1:j2], "+"))
        elif op == "insert":
            diff.append((corrected[j1:j2], "+"))
        elif op == "delete":
            diff.append((original[i1:i2], "-"))
    return diff


def format_diff_html(original: str, corrected: str) -> str:
    """diff를 HTML로 포맷."""
    diff = make_diff(original, corrected)
    parts = []
    for text, tag in diff:
        escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if tag == "+":
            parts.append(f'<span style="background:#bbffbb;text-decoration:none;">{escaped}</span>')
        elif tag == "-":
            parts.append(f'<span style="background:#ffbbbb;text-decoration:line-through;">{escaped}</span>')
        else:
            parts.append(escaped)
    return "".join(parts)
