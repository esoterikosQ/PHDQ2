import torch
from torch.utils.data import Dataset

class GecBltDataset(Dataset):
    """
    BLT(Byte Latent Transformer)를 위한 Prefix-LM 방식의 GEC 데이터셋 어댑터.
    TSV 파일(오류문 \t 교정문)을 읽어들여 UTF-8 바이트 시퀀스로 변환.
    [오류문 바이트들] + [SEP] + [교정문 바이트들] + [EOS] 형태로 구성.
    """
    def __init__(self, tsv_path, max_length=512, sep_byte=254, eos_byte=255, pad_byte=0):
        super().__init__()
        self.tsv_path = tsv_path
        self.max_length = max_length
        # BLT는 토크나이저 대신 256개 바이트(0~255)를 직접 사용.
        # 일반 UTF-8 바이트 외에 특수 토큰을 바이트 값의 끝부분에 매핑 (예시)
        self.sep_byte = sep_byte 
        self.eos_byte = eos_byte
        self.pad_byte = pad_byte
        
        self.data = self._load_tsv()

    def _load_tsv(self):
        samples = []
        with open(self.tsv_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip('\n').split('\t')
                if len(parts) == 2:
                    samples.append((parts[0], parts[1]))
        return samples

    def str_to_bytes(self, text):
        """문자열을 UTF-8 바이트 리스트로 변환"""
        return list(text.encode('utf-8'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]
        
        src_bytes = self.str_to_bytes(src_text)
        tgt_bytes = self.str_to_bytes(tgt_text)
        
        # 입력 시퀀스 구성: Src + [SEP] + Tgt + [EOS]
        input_seq = src_bytes + [self.sep_byte] + tgt_bytes + [self.eos_byte]
        
        # 라벨 구성: Causal LM을 위해 input과 동일하지만 loss masking을 적용
        # Source 부분과 SEP 토큰까지는 loss 계산에서 제외 (-100)
        labels = [-100] * (len(src_bytes) + 1) + tgt_bytes + [self.eos_byte]
        
        # Max length Truncation
        input_seq = input_seq[:self.max_length]
        labels = labels[:self.max_length]
        
        # Padding
        pad_len = self.max_length - len(input_seq)
        if pad_len > 0:
            input_seq.extend([self.pad_byte] * pad_len)
            labels.extend([-100] * pad_len)
            
        return {
            'input_ids': torch.tensor(input_seq, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

if __name__ == "__main__":
    # 간단한 테스트 스크립트
    # 더미 데이터 생성 후 확인
    import os
    if not os.path.exists("dummy.tsv"):
        with open("dummy.tsv", "w", encoding="utf-8") as f:
            f.write("안뇽하세요\t안녕하세요.\n이거슨 테스트\t이것은 테스트.\n")
            
    dataset = GecBltDataset("dummy.tsv", max_length=32)
    sample = dataset[0]
    
    print("Source \\t Target : 안뇽하세요 \\t 안녕하세요.")
    print(f"input_ids: {sample['input_ids']}")
    print(f"labels:    {sample['labels']}")
    
    os.remove("dummy.tsv")
