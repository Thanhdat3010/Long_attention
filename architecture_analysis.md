# Phân tích Chi tiết Kiến trúc LongAttention v2 & Đề xuất Cải tiến

Báo cáo này phân tích các điểm yếu cấu trúc trong phiên bản **LongAttention v2** hiện tại dựa trên mã nguồn thực tế tại [attention_core.py](file:///d:/Code/Long_attention/src/models/core/attention_core.py) và kết quả thực nghiệm từ [run_history.md](file:///d:/Code/Long_attention/run_history.md).

Mục tiêu cốt lõi của nghiên cứu này là vượt qua mô hình nền tảng **LED (Longformer Encoder-Decoder)** cả về hiệu năng (tốc độ huấn luyện) và độ chính xác (BLEU/ChrF++/COMET) để hướng tới một bài báo khoa học chất lượng cao gửi **ICLR (A*)**.

---

## 📊 So sánh Thực nghiệm Hiện tại (NMT task - CoCoDoc-MT-20k)

Theo ghi chép chạy thử nghiệm:

| Chỉ số | LED (v0) | LongAttention v2 (v1) | Đánh giá |
| :--- | :---: | :---: | :--- |
| **SacreBLEU (Test)** | **3.6824** | 3.5040 | ❌ LongAttention kém hơn **0.18 BLEU** |
| **ChrF++ (Test)** | **40.3282** | 40.0668 | ❌ LongAttention kém hơn **0.26** |
| **Tốc độ (Samples/s)** | **12.785** | 7.575 | ❌ LongAttention chậm hơn **1.7x** |
| **Thời gian chạy (s)** | **4693.0** | 7920.6 | ❌ LongAttention tốn thêm ~3200 giây |

---

## 🔍 Chi tiết 6 Điểm yếu Kiến trúc & Đề xuất Giải pháp

### 1. Tham số "Chết" (Dead Parameters): `A_encoded` và `affix_codebook`
* **Vị trí code**: 
  * Định nghĩa: [FunctionalDecomposer.affix_codebook](file:///d:/Code/Long_attention/src/models/core/attention_core.py#L65)
  * Tính toán: [FunctionalDecomposer.forward L81-L82](file:///d:/Code/Long_attention/src/models/core/attention_core.py#L81-L82)
  * Trích xuất: [LongAttention.forward L343](file:///d:/Code/Long_attention/src/models/core/attention_core.py#L343)
* **Hiện tượng**:
  Trong lớp `FunctionalDecomposer`, chúng ta phân tách `hidden_states` thành hai thành phần: `R_encoded` (Semantic Root) và `A_encoded` (Functional Affix). Lớp này khai báo `affix_codebook` dưới dạng một `nn.Linear(768, 768, bias=False)`. Tuy nhiên, trong hàm `LongAttention.forward`, giá trị `A_encoded` trả về bị bỏ qua hoàn toàn và không được sử dụng ở bất kỳ bước tính toán nào phía sau.
* **Hậu quả**:
  * **Lãng phí tham số**: Mỗi layer encoder lãng phí $768 \times 768 \approx 590K$ tham số. Với mô hình BART-base 6 layers, con số này là **~3.5 triệu tham số chết**.
  * **Lan truyền ngược vô ích**: Gradient vẫn được tính toán và truyền qua cổng `(1.0 - gate_score) * affix_features` nhưng không đóng góp vào hàm mất mát cuối cùng, gây lãng phí bộ nhớ và tài nguyên tính toán GPU.
* **Hướng giải quyết**:
  * **Cách A (Đơn giản nhất)**: Loại bỏ hoàn toàn `affix_codebook` và `A_encoded` nếu chúng ta chỉ muốn dùng `R_encoded` cho long-range branch.
  * **Cách B (Tận dụng lý thuyết)**: Dùng `A_encoded` để cộng gộp trực tiếp vào local attention hoặc skip connection, đảm bảo phần thông tin ngữ pháp/cú pháp không bị mất đi.

---

### 2. Sự Lệch Không Gian Biểu Diễn Giữa Query và Key (Q/K Space Mismatch)
* **Vị trí code**:
  * Tạo Key: [DependencyTypedGist L122-L141](file:///d:/Code/Long_attention/src/models/core/attention_core.py#L122-L141) nhận đầu vào là `R_encoded`.
  * Tạo Query: [TypedTopKRetrieval.forward L192](file:///d:/Code/Long_attention/src/models/core/attention_core.py#L192) chiếu trực tiếp từ `hidden_states` gốc.
* **Hiện tượng**:
  * **Key ($K$)**: Được tạo ra từ `R_encoded`, nghĩa là đã trải qua phép biến đổi `root_transform` và bị nhân với `gate_score` ($\in [0, 1]$).
  * **Query ($Q$)**: Được chiếu trực tiếp từ `hidden_states` gốc (không qua decomposer, không bị scale bởi `gate_score`).
* **Hậu quả**:
  * **Lệch miền giá trị (Scale Mismatch)**: Phép nhân dot-product giữa $Q$ (đầy đủ biên độ) và $K$ (bị thu nhỏ bởi `gate_score`) sẽ làm giảm đáng kể giá trị logits trước Softmax một cách phi lý.
  * **Lệch không gian đặc trưng (Representation Mismatch)**: Query tìm kiếm trên không gian của `hidden_states` gốc, trong khi Key đại diện cho không gian đã qua phân rã `Root`. Điều này làm giảm độ chính xác của cơ chế so khớp (retrieval alignment).
* **Hướng giải quyết**:
  * Đồng bộ hóa không gian biểu diễn: Chi chiếu Query từ `R_encoded` thay vì `hidden_states` gốc, hoặc thiết kế một phép biến đổi tương thích cho Query trước khi thực hiện attention.

---

### 3. Ngữ Cảnh Toàn Cục Yếu do Mean Pooling (Weak Global Context)
* **Vị trí code**: [LongAttention.forward L338](file:///d:/Code/Long_attention/src/models/core/attention_core.py#L338)
* **Hiện tượng**:
  Để tạo ngữ cảnh toàn cục phục vụ cho việc tính toán `gate_score` trong `FunctionalDecomposer`, chúng ta sử dụng phép trung bình đơn giản: `global_ctx = hidden_states.mean(dim=1, keepdim=True)`.
* **Hậu quả**:
  Khi độ dài chuỗi đầu vào $T$ rất lớn (1024, 2048 hoặc cao hơn), phép toán `mean` trên toàn bộ chiều sequence sẽ làm "loãng" (dilute) thông tin cực kỳ mạnh. Vector `global_ctx` lúc này sẽ tiến dần về một biểu diễn đồng nhất cho toàn bộ các token, làm cho `gate_proj_global` tính ra các logits gần như không đổi giữa các chuỗi khác nhau. Cơ chế gating toàn cục thực tế sẽ mất đi tính động (dynamic gating).
* **Hướng giải quyết**:
  * Dùng **Attention Pooling** hoặc **Segment-wise Pooling** (trung bình theo từng phân đoạn nhỏ hơn) thay vì mean pooling toàn bộ.
  * Hoặc sử dụng các token đặc biệt (như `<s>` / `[CLS]`) làm đại diện cho global context.

---

### 4. Cơ Chế Gating Kép Dư Thừa & Triệt Tiêu Nhau (Redundant Dual Gating)
* **Vị trí code**:
  * Necessity Gate ($g_i$): [NecessityGate L87-L106](file:///d:/Code/Long_attention/src/models/core/attention_core.py#L87-L106)
  * Decomposer Gate: [FunctionalDecomposer L53-L77](file:///d:/Code/Long_attention/src/models/core/attention_core.py#L53-L77)
* **Hiện tượng**:
  Cả hai cơ chế gating này đều dùng kiến trúc tương tự nhau (bottleneck MLP + SiLU + Sigmoid) trên cùng đầu vào `hidden_states` để cho ra các giá trị scalar nằm trong khoảng $[0, 1]$.
  * $g_i$ quyết định xem có lấy thông tin từ long-range branch hay không.
  * `gate_score` quyết định xem token đó có phải là Semantic Root để đưa vào long-range memory (`K_typed`, `V_typed`) hay không.
* **Hậu quả**:
  * **Xung đột triệt tiêu**: Nếu tại một bước, $g_i \approx 1.0$ (mô hình rất cần long-range) nhưng `gate_score` của các tokens đích lại $\approx 0.0$ (phân rã cho rằng chúng là Affix và lọc bỏ), thì $R_{encoded} \approx 0$, dẫn tới thông tin lưu trữ trong Key/Value dài hạn biến mất. Nhánh long-range lúc này sẽ trả về đầu ra trống rỗng bất chấp cổng $g_i$ đang mở tối đa.
  * **Trùng lặp tính toán**: Việc duy trì hai cơ chế gating độc lập nhưng có mục tiêu tương tự nhau làm tăng số lượng tham số và thời gian tính toán mà không mang lại sự khác biệt lớn về mặt biểu diễn.
* **Hướng giải quyết**:
  * Hợp nhất hoặc ràng buộc hai cổng này. Ví dụ: Dùng chung một hệ thống sinh gate logits đại diện cho "độ quan trọng ngữ nghĩa", sau đó phân bổ cho cả hai nhiệm vụ.

---

### 5. Diversity Loss Quá Khắc Nghiệt & Gây Nghẽn Hiệu Năng (Diversity Loss Bottleneck)
* **Vị trí code**: [TypedTopKRetrieval.forward L222-L236](file:///d:/Code/Long_attention/src/models/core/attention_core.py#L222-L236)
* **Hiện tượng**:
  * **Về mặt ngôn ngữ**: Loss này sử dụng khoảng cách Total Variation (TV Distance) để đẩy phân phối chú ý (attention weights) của các Dependency Types (Coreference, Lexical, Discourse) xa nhau nhất có thể (tiến về 1.0). Tuy nhiên, một token hoàn toàn có thể đồng thời đóng vai trò trong nhiều mối quan hệ (ví dụ: vừa là Coreference vừa là Lexical). Việc ép chúng phải phân tách hoàn toàn là không tự nhiên.
  * **Về mặt hiệu năng (Bottleneck)**: Đoạn code `indices = torch.randperm(T, device=hidden_states.device)[:num_samples]` tạo ra phép lấy mẫu ngẫu nhiên trên GPU, sau đó thực hiện các phép toán lập chỉ mục. Việc này ép buộc framework phải thực hiện đồng bộ hóa CPU-GPU (CPU-GPU synchronization) ở mỗi lượt forward, làm mất khả năng chạy song song tối ưu của GPU và kéo tụt Samples/sec từ 12.78 xuống 7.57.
* **Hậu quả**:
  * Làm mô hình chạy **chậm hơn 1.7 lần**.
  * Ép buộc phân tách quá mức làm giảm độ chính xác của các biểu diễn attention.
* **Hướng giải quyết**:
  * **Về hiệu năng**: Loại bỏ hoàn toàn `torch.randperm` và thay thế bằng việc cắt lát cố định (strided slicing) hoặc dùng generator không đồng bộ của PyTorch.
  * **Về lý thuyết**: Thay TV-distance bằng **Cosine Similarity Penalty** nhẹ nhàng hơn trên tầng chiếu (projection layers) của các loại dependency, hoặc dùng **Entropy Regularization** thay vì ép phân phối attention trực giao hoàn toàn.

---

### 6. Phép Cộng Hợp Nhất Thiếu Cân Bằng (Unbalanced Branch Merging)
* **Vị trí code**: [LongAttention.forward L360](file:///d:/Code/Long_attention/src/models/core/attention_core.py#L360): `combined_context = A_local + g_i * A_long`
* **Hiện tượng**:
  * `A_local` được sinh ra từ Local Sliding Window Attention với cửa sổ kích thước 512 (rất dày đặc, chứa nhiều thông tin).
  * `A_long` được sinh ra từ việc truy vấn Top-K (K=64) trên các gist phân rã (rất thưa thớt).
  * Việc cộng trực tiếp `A_local` với `g_i * A_long` không đi kèm bất kỳ cơ chế chuẩn hóa (normalization) hay nội suy (interpolation) nào.
* **Hậu quả**:
  Biên độ kích hoạt (activation magnitude) của hai nhánh này chênh lệch rất lớn. Khi huấn luyện sâu, nhánh Local có xu hướng lấn át hoàn toàn nhánh Long-range, hoặc việc cộng trực tiếp làm thay đổi phân phối giá trị kích hoạt đầu vào của lớp chiếu tiếp theo (`out_proj`), gây bất ổn định trong quá trình huấn luyện.
* **Hướng giải quyết**:
  * Sử dụng cơ chế nội suy tuyến tính (Gated Interpolation): 
    $$\text{combined\_context} = (1 - g_i) \cdot A_{\text{local}} + g_i \cdot A_{\text{long}}$$
  * Hoặc áp dụng một lớp Layer Normalization sau khi cộng hợp nhất để ổn định biên độ kích hoạt.

---

## 📈 Bảng Tổng Hợp Mức Độ Nghiêm Trọng & Thứ Tự Ưu Tiên

| Vấn đề | Phân loại | Mức độ nghiêm trọng | Độ khó sửa | Độ ưu tiên | Lợi ích kỳ vọng |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **5. Diversity Loss & GPU Bottleneck** | Hiệu năng / Lý thuyết | **🔴 Rất cao** | 🟢 Dễ | **1** | Tăng tốc độ huấn luyện lên ~1.5x, cải thiện độ hội tụ. |
| **1. Tham số chết (A_encoded)** | Tài nguyên / Tối ưu | **🔴 Cao** | 🟢 Dễ | **2** | Giảm ~3.5M tham số thừa, tăng tốc lan truyền ngược. |
| **2. Q/K Space Mismatch** | Toán học / Chất lượng | **🔴 Cao** | 🟡 Trung bình | **3** | Cải thiện độ chính xác phân phối chú ý dài hạn, tăng SacreBLEU. |
| **6. Gated Interpolation** | Tính ổn định | **🟡 Trung bình** | 🟢 Dễ | **4** | Ổn định gradient, giúp nhánh long-range đóng góp tốt hơn. |
| **4. Redundant Gating** | Thiết kế | **🟡 Trung bình** | 🟡 Trung bình | **5** | Đơn giản hóa kiến trúc, tránh xung đột triệt tiêu thông tin. |
| **3. Weak Global Context** | Lý thuyết | **🟡 Trung bình** | 🟡 Trung bình | **6** | Giúp cơ chế gating nhạy bén hơn với ngữ cảnh dài. |

---

## 💡 Đề xuất Kế hoạch Hành động Tiếp theo

Để đạt được mục tiêu vượt qua LED baseline nhanh nhất, chúng ta nên tiến hành theo 2 giai đoạn:

1. **Giai đoạn 1 (Sửa nhanh & Tối ưu hiệu năng - Quick Wins)**:
   * Loại bỏ `torch.randperm` trong Diversity Loss để giải phóng GPU bottleneck.
   * Loại bỏ các tham số chết của `A_encoded` và `affix_codebook` hoặc tích hợp chúng đúng cách.
   * Áp dụng công thức nội suy tuyến tính Gated Interpolation cho `combined_context`.
   * *Kiểm tra ngay tốc độ (samples/s) và BLEU score xem có cải thiện vượt bậc hay không.*

2. **Giai đoạn 2 (Tinh chỉnh Toán học & Lý thuyết - Core Quality)**:
   * Sửa lỗi lệch không gian Q/K bằng cách chiếu Query từ cùng một không gian đặc trưng.
   * Thiết kế lại cơ chế Global Context và Gating tối giản để tránh trùng lặp.
