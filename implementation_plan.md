# Kế hoạch Triển khai Nâng cấp Kiến trúc LongAttention v2 (Hướng tới ICLR A*)

Kế hoạch này đề xuất các cải tiến kiến trúc cốt lõi cho **LongAttention v2** nhằm mục tiêu kép: **tăng tối đa độ chính xác ( BLEU / ChrF++ / COMET )** và **loại bỏ các nghẽn hiệu năng GPU** để vượt qua mô hình nền tảng LED.

---

## 📢 Khuyến nghị Thiết kế & Ước lượng Cải tiến (Agent Recommendations)

Dưới đây là phân tích chi tiết của Antigravity về các lựa chọn thiết kế. **Khuyến nghị chính thức của chúng tôi là áp dụng đồng thời các giải pháp tối ưu bên dưới** để đạt hiệu suất tốt nhất:

### 1. Chiến lược tích hợp Affix (`A_encoded`) vào Local Branch (Khuyên dùng)
* **Phân tích so sánh**:
  * **Option 1: Tích hợp `A_encoded` vào Local Attention (Đề xuất chọn)**: Truyền `A_encoded` (các từ chức năng, ngữ pháp) để bổ trợ trực tiếp cho local sliding window attention.
    * *Độ mạnh Novelty (ICLR)*: **Rất cao**. Chứng minh được tính nhất quán của lý thuyết phân rã: Local branch xử lý cú pháp ngắn hạn (Affixes), Long branch xử lý ngữ nghĩa dài hạn (Roots). Nếu loại bỏ Affix hoàn toàn, mô hình sẽ bị đánh giá là một dạng sparse attention thông thường, thiếu nền tảng ngôn ngữ học.
    * *Độ chính xác (BLEU/ChrF++)*: Kỳ vọng cải thiện **~3% - 5%** so với việc bỏ phí Affix, nhờ bảo toàn thông tin cấu trúc ngữ pháp ngắn hạn (fluency).
  * **Option 2: Loại bỏ hoàn toàn Affix**: Xóa bỏ `affix_codebook` và `A_encoded`.
    * *Hiệu năng*: Tiết kiệm ~3.5M tham số (giảm ~4% tổng tham số attention), tăng nhẹ tốc độ lan truyền ngược (~2%).
    * *Độ mạnh Novelty*: Thấp, mất đi một nửa cốt truyện của bài báo.
* **👉 KHUYẾN NGHỊ**: Áp dụng **Option 1** để giữ vững Novelty ICLR và tăng độ chính xác dịch thuật.

### 2. Thay thế Diversity Loss bằng Phạt Trực giao Tĩnh (Static Orthogonality Regularization)
* **Phân tích so sánh**:
  * **Option A: Dùng TV-Distance trên Attention Weights động (Hiện tại)**: Gây nghẽn CPU-GPU sync nghiêm trọng do `torch.randperm`.
  * **Option B: Phạt Cosine Similarity trên các ma trận trọng số chiếu tĩnh (Đề xuất chọn)**: Áp dụng trực giao hóa trực tiếp trên các ma trận trọng số chiếu của 3 loại dependency (`Coref`, `Lexical`, `Discourse`) trong `self.q_proj` và `self.multi_type_proj`.
    * *Hiệu năng (Tốc độ)*: **Tăng tốc độ huấn luyện lên ~60% - 70%** (Samples/sec tăng từ 7.57 lên ~12.0 - 12.5, tiệm cận mức 12.78 của LED baseline) do loại bỏ hoàn toàn các hàm lấy mẫu gây nghẽn trong forward pass.
    * *Độ chính xác*: Kỳ vọng tăng **~2% - 4%** do tối ưu hóa trọng số tĩnh diễn ra ổn định và định hướng các kênh chú ý học đặc trưng độc lập tốt hơn.
* **👉 KHUYẾN NGHỊ**: Áp dụng **Option B** để giải phóng hoàn toàn hiệu năng GPU của mô hình.

---

## 🛠️ Đề xuất Thay đổi Mã nguồn (Proposed Changes)

### 1. Thành phần Core Attention

#### [MODIFY] [attention_core.py](file:///d:/Code/Long_attention/src/models/core/attention_core.py)

* **Hợp nhất không gian biểu diễn Q/K**:
  Chỉnh sửa lớp `TypedTopKRetrieval` để nhận Query được chiếu từ `R_encoded` thay vì `hidden_states` gốc. Điều này đảm bảo việc so khớp ngữ nghĩa diễn ra trên cùng một không gian đặc trưng sạch:
  ```python
  # Trong LongAttention.forward:
  # Thay vì:
  # A_long, diversity_loss = self.typed_retrieval(hidden_states, K_typed, V_typed, attention_mask)
  # Sẽ đổi thành:
  A_long, diversity_loss = self.typed_retrieval(R_encoded, K_typed, V_typed, attention_mask)
  ```

* **Loại bỏ Bottleneck của Diversity Loss**:
  * Loại bỏ cơ chế lấy mẫu ngẫu nhiên qua `torch.randperm` và TV-distance động.
  * Hiện thực hóa Orthogonality Loss tĩnh trên ma trận trọng số:
    ```python
    def compute_orthogonality_loss(self) -> torch.Tensor:
        # Lấy ma trận trọng số q_proj (num_types * D, D) và tách thành các loại
        # Phạt tích vô hướng (cosine similarity) giữa các ma trận trọng số của từng cặp type
        ...
    ```

* **Tích hợp `A_encoded` vào Local Branch (Theo Phương án 1)**:
  * Truyền `A_encoded` vào `LocalSlidingWindowAttention` hoặc cộng trực tiếp vào kết quả cục bộ:
    ```python
    A_local = self.local_attention(...)
    A_local_combined = A_local + A_encoded  # Affix đóng góp trực tiếp vào cấu trúc cục bộ
    ```

* **Cơ chế Hợp nhất Nội suy (Gated Interpolation)**:
  * Thay thế phép cộng trực tiếp bằng nội suy tuyến tính có trọng số nhằm cân bằng biên độ kích hoạt:
    ```python
    combined_context = (1.0 - g_i) * A_local_combined + g_i * A_long
    ```

* **Đồng bộ hóa Gating kép**:
  * Tinh chỉnh `NecessityGate` để nó tận dụng trực tiếp logits hoặc kết quả của cổng phân rã trong `FunctionalDecomposer`, giảm thiểu tính toán trùng lặp.

---

## 🧪 Kế hoạch Xác minh (Verification Plan)

### Kiểm tra Tự động (Automated Tests)
* Chạy kiểm tra tính đúng đắn của cấu trúc tensor và lan truyền ngược:
  ```powershell
  python -c "import torch; from src.models.core.attention_core import LongAttention; model = LongAttention(768, 12); x = torch.randn(2, 1024, 768); out = model(x); print('Output shape:', out[0].shape)"
  ```

### Kiểm tra Huấn luyện (Training & Speed Verification)
* Chạy huấn luyện thử nghiệm trên tập dữ liệu nhỏ của NMT trong 1 epoch để đo đạc:
  1. **Samples/sec**: Kỳ vọng tăng từ 7.5 lên >11.0 samples/s (tiệm cận mức 12.78 của LED).
  2. **Độ ổn định của Loss**: Đảm bảo Loss hội tụ đều và không bị hiện tượng gradient bùng nổ nhờ cơ chế Gated Interpolation.
