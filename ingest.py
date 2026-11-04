import os
import json
import pdfplumber

raw_dir = "data/raw"
output_file = "data/chunks.jsonl"

chunks = []
chunk_size = 400  # 每块大概 400 字
overlap = 50      # 重叠 50 字

for filename in os.listdir(raw_dir):
    if not filename.endswith(".pdf"):
        continue
    
    filepath = os.path.join(raw_dir, filename)
    print(f"正在处理: {filename}")
    
    try:
        with pdfplumber.open(filepath) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if not text:
                    continue
                
                # 按段落简单切分
                start = 0
                while start < len(text):
                    chunk_text = text[start:start + chunk_size]
                    if len(chunk_text.strip()) > 20:  # 过滤太短的块
                        chunks.append({
                            "id": f"{filename}_p{page_num}_{start}",
                            "text": chunk_text.strip(),
                            "source": filename,
                            "page": page_num
                        })
                    start += chunk_size - overlap
    except Exception as e:
        print(f"处理 {filename} 时出错: {e}")

# 保存为 JSONL
os.makedirs("data", exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

print(f"完成！共生成 {len(chunks)} 个文本块，保存在 {output_file}")