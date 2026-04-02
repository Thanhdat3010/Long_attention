import csv
import os
import re
from datasets import load_dataset
from tqdm import tqdm

def normalize(text):
    return re.sub(r"\s+", " ", text).strip()

def process_hotpotqa_gold(split_name, output_file, max_examples=20000):
    print(f"Loading HotpotQA '{split_name}' split...")
    ds = load_dataset("hotpot_qa", "distractor", split=split_name)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    saved_count = 0
    fallback_count = 0

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'context', 'question', 'answer_text', 'start_char', 'answers'])
        
        for i, item in enumerate(tqdm(ds, desc=f"Processing {split_name}")):
            if saved_count >= max_examples:
                break

            question = normalize(item['question'])
            answer_lower = normalize(item['answer']).lower()

            # 1. Thu thập các câu chứa Evidence (Supporting Facts)
            # Annotator của HotpotQA đã đánh dấu sẵn câu nào chứa manh mối.
            supp_facts = set()
            for sf_title, sf_sent_id in zip(item['supporting_facts']['title'], item['supporting_facts']['sent_id']):
                supp_facts.add(f"{sf_title}_{sf_sent_id}")

            long_context = ""
            best_start_char = -1
            best_answer_text = ""

            # 2. Xây dựng văn bản & Dò tìm chính xác
            for title, sentences in zip(item['context']['title'], item['context']['sentences']):
                # Thêm marker [PARA]
                para_marker = f"[PARA] {title}: "
                long_context += normalize(para_marker) + " "
                
                for sent_idx, sentence in enumerate(sentences):
                    sentence_norm = normalize(sentence)
                    
                    # Ghi nhận vị trí (offset) của câu hiện tại trong ngữ cảnh toàn cục
                    current_offset = len(long_context)
                    
                    # Nối câu vào ngữ cảnh
                    long_context += sentence_norm + " "
                    
                    # TÌM KIẾM CHUẨN A*: Chỉ tìm answer trong đúng câu Evidence!
                    if best_start_char == -1:
                        if f"{title}_{sent_idx}" in supp_facts:
                            sent_lower = sentence_norm.lower()
                            idx_in_sent = sent_lower.find(answer_lower)
                            if idx_in_sent != -1:
                                best_start_char = current_offset + idx_in_sent
                                best_answer_text = long_context[best_start_char : best_start_char + len(answer_lower)]

            long_context = long_context.strip()

            # 3. Fallback: Nếu vì lý do nào đó answer không nằm trong Evidence (Dataset noise)
            # Thì mới nhắm mắt dùng hàm find() toàn cục như code cũ của bạn
            if best_start_char == -1:
                idx_global = long_context.lower().find(answer_lower)
                if idx_global != -1:
                    best_start_char = idx_global
                    best_answer_text = long_context[best_start_char : best_start_char + len(answer_lower)]
                    fallback_count += 1
            
            # Nếu answer không xuất hiện ở đâu cả (bị lỗi dataset) -> Bỏ qua
            if best_start_char == -1:
                continue

            # 4. Ghi data (Sử dụng đúng 'id' nguyên bản của HotpotQA)
            writer.writerow([
                item['id'],
                long_context,
                question,
                best_answer_text,
                best_start_char,
                str([best_answer_text]) # List 1 phần tử an toàn
            ])
            saved_count += 1

    print(f"Saved {saved_count} samples to {output_file}")
    if fallback_count > 0:
        print(f"  Note: Used global fallback for {fallback_count} samples (answer found outside supporting facts).")
    print("-" * 50)

if __name__ == "__main__":
    process_hotpotqa_gold("train", "data/qa/train_hotpot.csv", max_examples=20000)
    process_hotpotqa_gold("validation", "data/qa/valid_hotpot.csv", max_examples=3000)
