# 한국어 GEC 데이터셋 (PHDQ)

이 디렉토리(`data/`)는 GEC 모델 학습 및 평가에 사용되는 데이터 세트를 보관하는 장소입니다.
**git에 커밋되지 않도록 `.gitignore`에 이미 `data/` 전체가 제외되어 있습니다.** (대용량 파일 방지)

## 파일 형식 (TSV)
`Standard_Korean_GEC` 코드(BART)의 데이터 가이드에 맞추어, **탭(Tab)으로 구분된 두 개의 컬럼**으로 만들어야 합니다.

```tsv
오류 문장 (Source)\t교정 문장 (Target)
```

예시 (`native_train.tsv`):
```text
안뇽하세요\t안녕하세요.
이거슨 테스트 문장 입니댜.\t이것은 테스트 문장입니다.
방가워\t반가워.
```

(주의: 컬럼 헤더 없이 첫 줄부터 바로 데이터가 와야 함. `dataset.py`의 `_read_docs` 참고)

## 명명 규칙
`scripts/train_bart.sh` 에서 편하게 데이터셋을 전환할 수 있도록 접두어를 통일합니다.

1. **Kor-Native** (원어민)
    - `native_train.tsv`
    - `native_dev.tsv`
    - `native_test.tsv`
2. **Kor-Lang8** (학습자)
    - `lang8_train.tsv`
    - `lang8_dev.tsv`
    - `lang8_test.tsv`
3. **Kor-Learner** (학습자 - 국립국어원 등 특수 코퍼스)
    - `learner_train.tsv`
...

## 사용법
1. 위 규칙에 맞게 파일을 준비해서 이 디렉토리에 넣습니다.
2. SLURM 클러스터에서 프로젝트 루트로 이동한 후 학습 스크립트를 실행합니다.
   ```bash
   sbatch scripts/train_bart.sh
   # (파일 내에서 DATASET_TYPE="native" 설정 확인)
   ```
