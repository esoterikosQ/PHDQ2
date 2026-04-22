# 한국어 GEC 프로젝트 종합 실행 계획 (Master Plan)

작성일: 2026-04-22
기준 저장소: esoterikosQ/PHDQ (현재 workspace)

---

## 1) 프로젝트 방향 요약

이 프로젝트는 3개 트랙을 병행하여, 최종적으로 "BLT를 결합한 한국어 GEC가 기존 KoBART GEC 대비 성능/효율에서 개선되는지"를 검증한다.

- Track 1 (Serving): 이미 학습된 GEC 모델을 RTX 5090 서버에서 웹 UI로 서비스
- Track 2 (Baseline): KoBART 기반 GEC 학습 재현 및 비교 기준 수립
- Track 3 (BLT-GEC): BLT 기반 GEC 구현 및 학습/평가

최종 산출물은 다음 3가지다.
1. 실제 동작하는 웹 데모 (교정 서비스)
2. 재현 가능한 베이스라인 학습 결과 (GLEU/M2)
3. BLT 결합 모델의 성능 비교 리포트

---

## 2) 현재 상태 (2026-04-22 기준)

완료:
- 3-트랙 구조 및 아키텍처 문서화 완료
- baseline 코드 마이그레이션(PL 2.x) 및 metric 이식 완료
- serving/architecture.md 생성 완료
- 프로젝트 관리 문서(SKILL, pipeline-reference, LOG) 정비 완료

미완료 (핵심 실행 단계):
- Track 1: infer.py, app.py, requirements.txt 구현 및 RTX 5090 배포
- Track 2: SLURM에서 데이터 준비/학습 실행/결과 확정
- Track 3: 데이터 어댑터 및 BLT 학습 루프 구현/실험
- 최종 비교 분석 및 보고서 정리

---

## 3) 목표 지표 (Success Criteria)

필수 목표:
- Serving: RTX 5090에서 웹 UI 응답형 GEC 서비스 정상 동작
- Baseline: 최소 1개 데이터셋(Kor-Native)에서 학습 재현 + GLEU/M2 확보
- BLT-GEC: 동일 평가셋에서 BLT 결과 산출 + Baseline과 정량 비교

권장 목표:
- 3개 데이터셋(Kor-Native, Kor-Lang8, Kor-Learner) 모두 결과 확보
- 오류 유형별(WS/SPELL/PUNCT) 분석 포함
- 추론 속도(지연 시간), 자원 사용량(GPU 메모리) 비교 포함

---

## 4) 전체 일정 (권장 12주 + 버퍼 2주)

프로젝트 기간:
- 본 실행: 12주 (2026-04-22 ~ 2026-07-14)
- 안정화/리스크 버퍼: 2주 (2026-07-15 ~ 2026-07-28)

### 4-1. 마일스톤

| 마일스톤 | 기간 | 완료 기준 |
|---|---|---|
| M1. Serving MVP | W1-W2 (4/22-5/5) | 웹 UI에서 단일 문장 교정 가능 |
| M2. Baseline 1차 재현 | W2-W5 (4/29-5/26) | Kor-Native 학습/평가 수치 확보 |
| M3. Baseline 확장 | W5-W6 (5/20-6/2) | 3개 데이터셋 결과 확보 |
| M4. BLT-GEC 1차 구현 | W4-W7 (5/13-6/9) | Prefix-LM 학습 코드 실행 성공 |
| M5. BLT-GEC 실험 완료 | W7-W10 (6/3-6/30) | BLT 결과 수치 확보 |
| M6. 최종 비교/보고 | W10-W12 (6/24-7/14) | 비교표, 분석, 결론 문서 완료 |
| Buffer | W13-W14 (7/15-7/28) | 재실험/튜닝/배포 안정화 |

### 4-2. 주차별 실행 로드맵

| 주차 | 날짜 | 핵심 작업 | 주 실행 머신 |
|---|---|---|---|
| W1 | 4/22-4/28 | Serving 코드 스캐폴딩(infer/app), 체크포인트 경로 확정 | Mac, Ubuntu |
| W2 | 4/29-5/5 | Serving MVP 배포 + Baseline 데이터 준비 | Ubuntu, SLURM |
| W3 | 5/6-5/12 | Baseline Kor-Native 학습 1차 실행 | SLURM |
| W4 | 5/13-5/19 | Baseline 튜닝 + BLT 데이터 어댑터 구현 시작 | SLURM, Mac |
| W5 | 5/20-5/26 | Baseline 수치 확정(최소 1~2셋) | SLURM |
| W6 | 5/27-6/2 | Baseline 3셋 확장 완료 + BLT 학습 루프 연결 | SLURM |
| W7 | 6/3-6/9 | BLT-GEC 학습 1차 성공(작은 설정) | SLURM |
| W8 | 6/10-6/16 | BLT 본 실험 1차(하이퍼파라미터 스윕) | SLURM |
| W9 | 6/17-6/23 | BLT 본 실험 2차 + 실패 케이스 분석 | SLURM |
| W10 | 6/24-6/30 | BLT 결과 확정 + Serving에 A/B 비교 탭 설계 | SLURM, Ubuntu |
| W11 | 7/1-7/7 | 비교 분석(성능/속도/오류유형) | Mac |
| W12 | 7/8-7/14 | 최종 보고서/발표 자료 정리 | Mac |

---

## 5) 트랙별 상세 작업과 소요 기간

## Track 1: Serving (기학습 GEC 웹 서비스)

목표:
- 기학습 KoBART GEC를 RTX 5090 서버에서 실서비스 형태로 제공

예상 기간: 2주 (MVP), 확장 1주

세부 작업:
1. 체크포인트 확보/검증 (1-2일)
   - 공개 체크포인트 링크 검증
   - 로딩 포맷(.ckpt/.bin) 확인
2. 추론 엔진 구현 infer.py (2일)
   - model/tokenizer 로드
   - generate 파라미터 고정(num_beams=4 등)
   - 배치/단일 추론 API
3. 웹 UI 구현 app.py (2-3일)
   - 입력/출력
   - diff 하이라이트
   - 에러 핸들링
4. 서버 배포/접속 테스트 (2일)
   - Ubuntu 서비스 실행
   - 외부 접속/방화벽/포트 점검
5. 안정화 (2일)
   - OOM 방지
   - timeout/retry

산출물:
- serving/infer.py
- serving/app.py
- serving/requirements.txt
- 동작 스크린샷/접속 URL(내부)

리스크:
- 공개 체크포인트 미제공 가능성
대응:
- Track 2 결과 체크포인트를 serving에 투입

---

## Track 2: Baseline (KoBART GEC 학습 재현)

목표:
- 비교 기준이 되는 신뢰 가능한 baseline 점수 확보

예상 기간: 4~5주

세부 작업:
1. 데이터 준비 (4-5일)
   - Kor-Native/Lang8/Learner split 정리
   - 파일 경로 표준화
2. 학습 스크립트 SLURM 연동 (2일)
   - scripts/train_bart.sh
   - 로그/체크포인트 경로 통일
3. Kor-Native 1차 학습/평가 (4-5일)
   - 최소 1회 full train
   - GLEU/M2 산출
4. 하이퍼파라미터 조정 (3-4일)
   - lr, batch, max_epochs 조정
5. 3개 데이터셋 확장 실험 (1-2주)
   - Lang8/Learner/Union 추가

산출물:
- baseline 학습 로그
- 체크포인트
- 데이터셋별 결과표 (GLEU, M2 P/R/F0.5)

리스크:
- 구버전 코드 유산으로 인한 평가 불안정
대응:
- baseline/metric 기준으로 평가 파이프라인 고정

---

## Track 3: BLT-GEC (BLT 결합 모델)

목표:
- Prefix-LM 방식 BLT-GEC 구현 및 baseline 대비 비교

예상 기간: 5~6주

세부 작업:
1. 데이터 어댑터 구현 (1주)
   - TSV -> UTF-8 bytes
   - [BOS][src][SEP][tgt][EOS]
   - loss mask(SEP 이후만)
2. BLT 학습 코드 결합 (1주)
   - bytelatent train loop 연결
   - patching(entropy) 통합
3. 소규모 스모크 실험 (3-4일)
   - 작은 subset으로 overfit 확인
4. 본 학습 실험 (2주)
   - lr, batch, max_seqlen, grad_acc 스윕
5. 평가/분석 (1주)
   - Baseline과 동일 지표 산출
   - 오류 유형별 분석

산출물:
- blt_gec 학습 코드
- 실험 로그/체크포인트
- 비교표 (Baseline vs BLT)

리스크:
- BLT 가중치 접근 승인/학습 비용/패칭 민감도
대응:
- blt-1b로 시작, 실험 수를 단계적으로 확장

---

## 6) 의존관계와 병렬화 전략

핵심 의존관계:
- Track 1은 체크포인트 확보에 Track 2 결과를 fallback으로 사용 가능
- Track 3 평가는 Track 2 지표 파이프라인이 기준

병렬화 권장:
- W1-W2: Track 1 + Track 2 준비 병행
- W4-W6: Track 2 실험 + Track 3 구현 병행
- W10 이후: 분석/문서화 + Serving 개선 병행

---

## 7) 머신별 역할 분담(실행 기준)

Mac:
- 코드 작성, PR/commit, 문서화, 결과 정리

SLURM:
- Baseline/BLT 학습 및 대규모 실험
- 결과 파일 생성 및 push

Ubuntu RTX 5090:
- Serving 앱 배포/운영
- 데모 검증

운영 원칙:
- 코드 변경은 Mac에서 수행 후 push
- SLURM/Ubuntu는 실행 전 항상 pull

---

## 8) 주간 운영 체크리스트

매주 월요일:
- 이번 주 실험 목표 1~2개 고정
- 필요한 자원(SLURM GPU, 서버 포트) 사전 확보

매일 종료 전:
- LOG.md 업데이트
- 결과/실패 원인 기록
- 다음 액션 1개 명시

매주 금요일:
- 주간 결과 요약(성공/실패/다음 실험)
- 일정 지연 시 범위 조정(실험 개수 축소 등)

---

## 9) 최종 결과물 패키지 (완료 기준)

1. 서비스 결과물
- RTX 5090에서 실행 가능한 GEC 웹 UI

2. 연구 결과물
- Baseline 재현 결과표
- BLT-GEC 결과표
- 성능 비교/해석 문서

3. 재현성 결과물
- 실행 스크립트
- 환경 requirements
- 실험 로그/설정 기록

---

## 10) 즉시 실행 To-do (다음 2주)

1. serving/infer.py, serving/app.py, serving/requirements.txt 구현 (3~4일)
2. 공개 체크포인트 확보 여부 확정 (1일)
3. SLURM baseline 1차 실행 (3~5일)
4. 결과 로그 템플릿 확정 및 주간 리뷰 루틴 시작 (1일)

예상 성과(2주 후):
- 웹 서비스 MVP 동작
- baseline 첫 수치 확보
- BLT 구현 착수 가능한 상태
