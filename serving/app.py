"""
한국어 GEC 웹 UI (Gradio).
사용: python app.py [--model MODEL_NAME] [--port PORT] [--share]
"""
import argparse

import gradio as gr

from infer import GecInferenceEngine, format_diff_html

# ── 전역 엔진 (서버 시작 시 1회 로드) ──────────────────────────
engine: GecInferenceEngine | None = None


def correct_single(text: str) -> tuple[str, str]:
    """단일 문장 교정 → (교정 결과, diff HTML)"""
    corrected = engine.correct(text)
    if corrected == text:
        diff_html = "<em>변경 사항 없음</em>"
    else:
        diff_html = format_diff_html(text, corrected)
    return corrected, diff_html


def correct_multi(text: str) -> tuple[str, str]:
    """여러 줄 교정 (줄 단위) → (교정 결과, diff HTML)"""
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return "", ""

    corrected_lines = engine.correct_batch(lines)
    diff_parts = []
    for orig, corr in zip(lines, corrected_lines):
        if orig == corr:
            diff_parts.append(f"<p>{orig.replace('<', '&lt;')}</p>")
        else:
            diff_parts.append(f"<p>{format_diff_html(orig, corr)}</p>")

    return "\n".join(corrected_lines), "\n".join(diff_parts)


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="한국어 문법 교정기 (Korean GEC)",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# 한국어 문법 교정기\nKoBART 기반 GEC 모델 (`Soyoung97/gec_kr`)")

        with gr.Tab("단일 문장"):
            with gr.Row():
                with gr.Column():
                    input_single = gr.Textbox(
                        label="입력 (오류 문장)",
                        placeholder="예: 한국어는어렵다.",
                        lines=3,
                    )
                    btn_single = gr.Button("교정하기", variant="primary")
                with gr.Column():
                    output_single = gr.Textbox(label="교정 결과", lines=3, interactive=False)
                    diff_single = gr.HTML(label="변경 사항")

            btn_single.click(
                correct_single,
                inputs=input_single,
                outputs=[output_single, diff_single],
            )

            gr.Examples(
                examples=[
                    "한국어는어렵다.",
                    "나는 학교에갔다",
                    "오늘날시가 좋습니다.",
                    "그녀는 영화를봤다",
                ],
                inputs=input_single,
            )

        with gr.Tab("여러 문장"):
            with gr.Row():
                with gr.Column():
                    input_multi = gr.Textbox(
                        label="입력 (줄 단위)",
                        placeholder="한 줄에 한 문장씩 입력하세요.",
                        lines=8,
                    )
                    btn_multi = gr.Button("전체 교정하기", variant="primary")
                with gr.Column():
                    output_multi = gr.Textbox(label="교정 결과", lines=8, interactive=False)
                    diff_multi = gr.HTML(label="변경 사항")

            btn_multi.click(
                correct_multi,
                inputs=input_multi,
                outputs=[output_multi, diff_multi],
            )

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="Korean GEC Serving")
    parser.add_argument("--model", type=str, default="Soyoung97/gec_kr",
                        help="HuggingFace 모델 이름 또는 로컬 체크포인트 경로")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cpu (기본: auto)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Gradio 공유 링크 생성")
    return parser.parse_args()


def main():
    global engine
    args = parse_args()
    engine = GecInferenceEngine(model_name=args.model, device=args.device)
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
