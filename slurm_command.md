# GPU 자원 확인

```bash
# 전체 파티션의 자원 상황 일람
sinfo

# 특정 파티션의 자원 상황
sinfo -p [파티션] -Nel


```

# 작업 확인

```bash
# 사용자별 작업 목록
squeue -u [ID]

# 작업이 진행중인 경우에는 gpu 노드에 접속해서 작업 상황을 확인할 수 있음
ssh gpu[GPU 노드 번호]
nvidia-smi -l 2

# 작업 삭제
scancel [작업번호]

```

# 작업 요청

```bash
# 작업 요청 방법
sbatch [쉘스크립트 파일명]

# 플래그 설명
sbatch -J [작업명] --time [최대작업시간] -o [작업로그파일명 (1출력)] -e [에러로그 파일명(2출력)] -p [파티션 이름] --comment [애플리케이션 이름] --nodes [작업 노드 수] --gres=gpu:[GPU 요청 수] --cpus-per-task=[CPU 요청 수] 
```