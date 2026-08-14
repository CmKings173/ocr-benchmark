# OCR Benchmark Suite — Requirements Questionnaire

**Mục đích:** khóa phạm vi và các giả định trước khi thiết kế chi tiết hoặc viết code.

## Cách trả lời

- Với câu hỏi một lựa chọn, đổi [ ] thành [x] ở đúng một đáp án.
- Với câu hỏi cho phép nhiều lựa chọn, có thể chọn nhiều đáp án.
- Điền phần “Khác / ghi chú” khi đáp án có sẵn không phù hợp.
- Có thể giữ nguyên đáp án khuyến nghị nếu chưa có quyết định riêng.

## Các quyết định đã ghi nhận

| Mục | Đã chọn |
|---|---|
| Mô hình vận hành V1 | Một người vận hành, một máy NVIDIA, chạy benchmark tuần tự |
| Phạm vi V1 | Toàn bộ Tier A: PP-OCRv6, GLM-OCR, PaddleOCR-VL, MonkeyOCRv2 |
| Máy tham chiếu | ASUS Ascent GX10 bản 4TB |
| Hệ điều hành | NVIDIA DGX OS / Ubuntu 24.04 mặc định |
| Dataset cardinality V1 | Chính xác một tem vật lý trên mỗi ảnh; validator từ chối dataset vi phạm |

> Cần xác minh lại thông tin runtime thực tế bằng inspect_environment; 4TB là SSD, không phải VRAM. GX10 dùng unified memory nên report phải phân biệt rõ unified memory với VRAM của GPU rời.

## Decision Log

| ID | Quyết định | Phương án đã cân nhắc | Lý do |
|---|---|---|---|
| D-001 | Một người vận hành, một máy NVIDIA, chạy tuần tự | Multi-user/distributed scheduler | Phù hợp quy mô V1 và ưu tiên reproducibility |
| D-002 | V1 triển khai toàn bộ Tier A | Một vertical slice PP-OCRv6 trước; toàn bộ đặc tả ngay V1 | Cần so sánh các họ OCR/VLM chính ngay trong delivery đầu |
| D-003 | ASUS Ascent GX10 4TB là máy tham chiếu | Không có máy chuẩn; nhiều loại GPU ngang hàng | Kết quả leaderboard phải cùng phần cứng và môi trường |
| D-004 | V1 chỉ benchmark một tem vật lý trên mỗi ảnh | Ghép nhiều tem không có annotation; multi-label ngay V1 | Chưa có bbox/region annotation hoặc unique matching key đáng tin cậy |
| D-005 | Memory production gate là cấu hình tùy chọn; không hard-code 16GB | Giới hạn cố định 16GB | 16GB chỉ là ví dụ và giới hạn thực tế có thể thay đổi theo target deployment |
| D-006 | Report riêng OCR/text quality và end-to-end field extraction | Gộp tất cả vào một accuracy score | Tránh extraction strategy che lấp chất lượng OCR hoặc tạo lợi thế không công bằng |
| D-007 | 100 ảnh chỉ dùng exploratory; production acceptance dùng point estimate và 95% confidence interval | Enforce gate trực tiếp trên 100 ảnh | Dataset nhỏ không đủ độ phân giải để chứng minh mục tiêu 99,5–99,9% |
| D-008 | Production gates và score weights nằm trong config; model fail gate không được score bù | Hard-code threshold/weights; chỉ xếp hạng bằng composite score | SLA thay đổi theo khách hàng và raw metrics phải giữ vai trò quyết định |
| D-009 | Recommendation V1 chỉ có giá trị chắc chắn cho GX10 | Suy rộng sang server giá thấp/enterprise chưa benchmark | Không đưa ra kết luận phần cứng thiếu dữ liệu thực nghiệm |
| D-010 | Barcode baseline gồm OpenCV và ZXing-C++; CJK được tag riêng | Chỉ một barcode engine; gộp toàn bộ CJK | Tăng coverage barcode và tránh trộn các hệ chữ khác nhau |
| D-011 | Freeze prompt, regex và extraction config trước khi chạy holdout | Tune trực tiếp trên customer holdout | Ngăn leakage và giữ leaderboard có ý nghĩa |
| D-012 | Cost metrics là N/A khi thiếu hardware/electricity inputs | Tự ước lượng ngầm | Không biến estimate chưa có căn cứ thành fact |
| D-013 | Không có deadline cố định; triển khai theo phase nhưng delivery đầu vẫn bao gồm Tier A | Chốt deadline giả định | Chưa có ràng buộc thời gian được cung cấp |
| D-014 | Orchestrator + subprocess worker; chỉ container hóa worker khi dependency xung đột | Một Python process; container cho mọi model ngay V1 | Cân bằng isolation, reproducibility và độ phức tạp trên GX10 |
| D-015 | Đo latency ở cả worker và orchestrator | Chỉ đo một phía | Tách inference stages khỏi IPC, startup và pipeline overhead |
| D-016 | Accuracy pass và performance pass là hai dataset execution độc lập | Dùng repeated performance inference để tăng accuracy sample | Giữ accuracy không bị bias bởi performance repetitions |
| D-017 | Structural normalization được phép; value normalization không được dùng cho strict exact match | Chuẩn hóa giá trị trước khi tính accuracy | Bảo vệ SKU/serial/barcode exactness cho ERP |
| D-018 | Tách model benchmark, barcode benchmark và system benchmark | Gộp OCR model result với barcode decoder result | Phân biệt chất lượng model với chất lượng pipeline production |

---

## 1. Mục tiêu và người dùng

### Q1. Quyết định chính mà benchmark phải hỗ trợ là gì? (một lựa chọn)

- [ ] A. Chọn model OCR tốt nhất trên dataset khách hàng
- [ ] B. Chọn cấu hình deploy production và SLA
- [x] C. Cả A và B (khuyến nghị)

### Q2. Ai là người đọc và phê duyệt kết quả? (có thể chọn nhiều)

- [x] A. Kỹ sư ML/MLOps
- [x] B. Backend/Platform engineer
- [ ] C. Người phụ trách khách hàng/ERP
- [ ] D. Quản lý kỹ thuật
- [ ] E. Khác / ghi chú: ____________________

### Q3. V1 có deadline hoặc giới hạn thời gian nào không?

Answer: Chưa có deadline cố định. Triển khai theo phase; delivery đầu vẫn phải bao gồm toàn bộ adapter Tier A.

### Q4. Ai sẽ sở hữu việc bảo trì adapter và cập nhật model?

- [x] A. Một owner duy nhất (khuyến nghị cho V1)
- [ ] B. Nhóm ML/MLOps
- [ ] C. Nhóm Platform/backend
- [ ] D. Khác / ghi chú: ____________________

---

## 2. Dataset và ground truth

### Q5. Quy mô dataset chính thức của V1 là gì? (một lựa chọn)

- [x] A. 100 ảnh ban đầu, thiết kế mở rộng lên 1.000–10.000+
- [ ] B. 1.000–10.000 ảnh ngay từ V1
- [ ] C. Khác: ____________________

### Q6. Có tách dataset thành các tập riêng không? (một lựa chọn)

- [x] A. Chỉ benchmark một tập customer holdout, không dùng để tune model (khuyến nghị)
- [ ] B. dev để phát triển và holdout để xếp hạng
- [ ] C. train/dev/test đầy đủ
- [ ] D. Chưa quyết định

Prompt, regex và extraction config phải được version hóa và freeze trước khi chạy customer holdout. Không dùng kết quả holdout để tune rồi chạy lại như một leaderboard độc lập.

### Q7. Ground truth có bounding box cho text/barcode không?

- [ ] A. Có bbox cho toàn bộ hoặc phần lớn mẫu
- [x] B. Chỉ có field/text, chưa có bbox (khi đó detection metric là N/A)
- [x] C. Có kế hoạch bổ sung sau

### Q8. Danh sách field chuẩn của tem là gì?

- [ ] A. customer, po_number, sku, lot, quantity, unit, serial, barcode, date (khuyến nghị làm baseline)
- [x] B. Danh sách khác: Dynamic field schema theo label_type/sample.

### Q9. Field nào là required để tính full_label_exact_match?

Required và critical fields được định nghĩa theo label_type hoặc từng sample, không dùng một danh sách global. Production gates tham chiếu tập critical fields đã khai báo trong schema tương ứng.

### Q10. Có cho phép một ảnh có nhiều tem hoặc nhiều giá trị cùng field không?

- [x] A. V1 chỉ cho phép một tem vật lý trên mỗi ảnh; dataset validator phải flag validation error và từ chối benchmark nếu phát hiện nhiều tem
- [ ] B. Có, cần schema danh sách/instance
- [ ] C. Chưa quyết định

Schema có thể giữ khả năng mở rộng, nhưng multi-label evaluation là phase sau và chỉ được bật khi có bbox/region annotation hoặc unique matching key đáng tin cậy.

### Q11. Các tag điều kiện ảnh V1 gồm những gì?

- [x] A. clear, blur, rotated, reflection, dark, small_text, damaged, long_distance, vietnamese, mixed (khuyến nghị)
- [ ] B. Dùng danh sách khác: ______________________________________

### Q12. Ngôn ngữ/chữ viết cần benchmark?

- [x] A. Tiếng Việt + Latin/English + số/ký hiệu
- [x] B. Thêm chữ viết khác: Simplified Chinese, Traditional Chinese và Japanese được gắn tag riêng; chỉ tính metric khi dataset thực tế có ngôn ngữ đó
- [ ] C. Chỉ Latin/English + số

### Q13. Barcode/QR có ground truth exact value và loại symbology không?

- [x] A. Có cả value và loại mã
- [ ] B. Chỉ có value
- [ ] C. Chưa có barcode ground truth

### Q14. Chính sách dữ liệu khách hàng là gì?

- [x] A. Chạy offline hoàn toàn, không upload ra ngoài (khuyến nghị)
- [ ] B. Cho phép gọi endpoint nội bộ
- [ ] C. Cho phép cloud/API bên ngoài khi được phê duyệt
- [ ] D. Khác / ghi chú: ____________________

---

## 3. Định nghĩa metric và protocol

### Q15. Normalization nào được phép trước exact match? (có thể chọn nhiều)

- [x] A. Trim leading/trailing whitespace
- [x] B. Chuẩn hóa line ending và khoảng trắng liên tiếp
- [ ] C. Case-fold (ABC và abc coi như nhau)
- [x] D. Chuẩn hóa Unicode
- [x] E. Không sửa ký tự/hyphen/digit-letter confusion (bắt buộc cho SKU/serial/barcode)
- [ ] F. Quy tắc field-specific khác: ____________________

### Q16. Nếu model không trả được required field thì tính thế nào?

- [x] A. Fail field và fail full label (khuyến nghị)
- [ ] B. Bỏ qua field thiếu trong full label
- [ ] C. Tách riêng unreadable khỏi wrong

### Q17. Latency end-to-end bao gồm những bước nào? (có thể chọn nhiều)

- [x] A. Đọc ảnh
- [x] B. Preprocess
- [x] C. Barcode decode
- [x] D. OCR inference
- [x] E. Field extraction/validation
- [ ] F. Ghi output/report
- [x] G. Report cả hai: model_only và end_to_end (khuyến nghị)

Accuracy cũng được report thành hai lớp: (1) OCR/text quality khi output có thể so sánh và (2) end-to-end field extraction theo một versioned extraction contract. Model-specific extraction strategy phải được ghi vào metadata.

### Q18. Throughput/concurrency cần benchmark mức nào?

- [x] A. Batch 1,4,8; concurrency 1,2,4,8 (khuyến nghị phù hợp V1)
- [ ] B. Chỉ batch 1, concurrency 1
- [ ] C. Mức khác: __________________________________

### Q19. Warmup và số lần lặp mặc định?

accuracy:
  iterations_per_image: 1

performance:
  warmup_iterations: 5
  min_iterations: 100
  repetitions: 3

### Q20. Retry và timeout được xử lý thế nào?

- timeout_seconds: 60 (đề xuất: 60)
- max_retries: ____0__ (đề xuất: 0 cho accuracy run; retry riêng trong reliability test)

### Q21. Confidence calibration có bắt buộc trong V1 không?

- [ ] A. Có nếu adapter cung cấp confidence và dataset đủ lớn
- [ ] B. Có bắt buộc cho mọi model
- [x] C. Optional; thiếu confidence thì ghi N/A (khuyến nghị)

### Q22. Barcode engine baseline nào được phép dùng? (có thể chọn nhiều)

- [x] A. OpenCV barcode
- [ ] B. ZBar
- [x] C. ZXing-C++
- [x] D. So sánh nhiều engine
- [ ] E. Khác: ____________________

---

## 4. Model và adapter

### Q23. Nếu official implementation không chạy được trên GX10 thì xử lý thế nào?

- [x] A. Ghi NOT_SUPPORTED/DEPENDENCY_ERROR và tiếp tục model khác (khuyến nghị)
- [ ] B. Cho phép dùng backend/fork thay thế nếu ghi rõ provenance
- [ ] C. Dừng V1 đến khi model chạy được

### Q24. Tier B/C có đưa vào leaderboard không?

- [x] A. Có kết quả hợp lệ thì đưa; status không hợp lệ thì không rank (khuyến nghị)
- [ ] B. Chỉ Tier A được rank
- [x] C. Tách leaderboard theo tier

Diễn giải: chỉ result hợp lệ mới được rank và leaderboard được tách theo tier; các status SKIPPED/NOT_SUPPORTED/DEPENDENCY_ERROR/OOM/TIMEOUT vẫn xuất hiện trong report nhưng không có rank.

### Q25. Backend được phép cho từng model là gì?

- [x] A. Official backend trước; vLLM khi official hỗ trợ
- [ ] B. Bất kỳ backend tương thích nào, miễn pin version và ghi metadata
- [ ] C. Chỉ backend đã được phê duyệt: ____________________

### Q26. Quantization cần benchmark ngay trong V1 không?

- [x] A. Chỉ native precision trước; quantization ở phase sau (khuyến nghị)
- [ ] B. BF16/FP16 bắt buộc
- [ ] C. Thêm INT8/INT4 nếu backend hỗ trợ

### Q27. Model revision và license metadata cần mức kiểm soát nào?

- [x] A. Bắt buộc lưu repository, model ID, exact revision, license, source URL, ngày xác minh (khuyến nghị)
- [ ] B. Chỉ lưu model ID/version
- [ ] C. Khác: ____________________

### Q28. Mistral OCR hoặc model endpoint riêng có cần cấu hình trong V1 không?

- [x] A. Chưa cần; adapter và SKIPPED_NOT_CONFIGURED là đủ
- [ ] B. Có endpoint nội bộ: ____________________
- [ ] C. Có private container: ____________________

### Q29. Chính sách cache/download model?

- [x] A. Cache local mount persistent; có lệnh pre-download; benchmark offline sau đó (khuyến nghị)
- [ ] B. Download tự động mỗi lần chạy
- [ ] C. Cache ở thư mục khác: ____________________

---

## 5. Runtime trên ASUS Ascent GX10

### Q30. Xác nhận thông số thực tế từ máy

Điền sau khi chạy lệnh kiểm tra môi trường:

- OS: ___________Ubuntu 24.04.4 LTS _________
- Architecture: ______aarch64 (ARM64)______________
- CPU: ______20 cores
     - ARM Cortex-X925
     - ARM Cortex-A725______________
- Unified memory: ______121 GiB usable (~128 GB physical)______________
- GPU/GB10: ______NVIDIA GB10______________
- CUDA/driver: _____ - NVIDIA Driver: 580.173.02
     - CUDA supported by driver: 13.0_______________
- Python: ____3.12.3________________

### Q31. Cách report memory trên unified-memory architecture?

- [x] A. Report unified memory used/peak; thêm GPU counters nếu runtime cung cấp; không gọi đó là VRAM rời (khuyến nghị)
- [ ] B. Chỉ report tổng RAM
- [ ] C. Quy ước khác: ____________________

### Q32. Process isolation cho model được chọn thế nào?

- [x] A. Mỗi model một subprocess; container hóa khi dependency xung đột (khuyến nghị)
- [ ] B. Tất cả model trong một Python process
- [ ] C. Mỗi model một container ngay từ V1

### Q33. Mức tương thích ARM64/GX10 cần cam kết?

- [x] A. Chỉ đánh dấu model chạy thật trên GX10 là supported; không giả lập compatibility (khuyến nghị)
- [ ] B. Bắt buộc mọi Tier A phải ARM64-native
- [ ] C. Cho phép chạy model trên máy x86 khác và gắn nhãn kết quả riêng

### Q34. Có cần hỗ trợ hai GX10 hoặc server NVIDIA rời trong phase sau không?

- [x] A. Chưa cần trong V1, nhưng không khóa schema/result
- [ ] B. Có roadmap multi-GX10
- [ ] C. Có roadmap server x86 + discrete GPU
- [ ] D. Cả B và C

Recommendation V1 chỉ được khẳng định cho máy tham chiếu GX10. Hardware khác phải có benchmark run riêng trước khi đưa ra kết luận production.

---

## 6. Pipeline, fallback và service

### Q35. Fallback pipeline có thuộc V1 không?

- [ ] A. Có benchmark riêng sau khi single-model benchmark hoàn tất
- [x] B. Chưa cần V1; giữ interface để thêm sau (khuyến nghị)
- [ ] C. Bắt buộc trong V1: ____________________

### Q36. Rule accept/fallback dựa trên gì? (có thể chọn nhiều)

- [x] A. Field exact validation
- [x] B. Confidence threshold
- [x] C. Barcode decode failure
- [x] D. Model timeout/error
- [ ] E. Rule khác: ____________________

### Q37. Service mode có nằm trong V1 không?

- [x] A. CLI benchmark trước; giữ adapter có thể tái sử dụng cho FastAPI sau (khuyến nghị)
- [ ] B. Cần luôn POST /ocr trong V1
- [ ] C. Chưa có nhu cầu service

---

## 7. Ranking, báo cáo và acceptance

### Q38. Ai sở hữu score weights và production gates?

- [x] A. Cấu hình YAML, team kỹ thuật chỉnh được, không hard-code (khuyến nghị)
- [ ] B. Một bộ weights cố định do ____________________ phê duyệt
- [ ] C. Chưa có weights; chỉ report metric raw trước

### Q39. Production gates ban đầu là gì?

```yaml
production_gates:
  # Các giá trị dưới đây là default minh họa và được override theo SLA/customer.
  full_label_accuracy_min: 0.995
  critical_field_accuracy_min: 0.999
  failure_rate_max: 0.001
  p95_latency_max_ms: 500
  p99_latency_max_ms: 1000
  # Optional and deployment-specific. null means the memory gate is disabled.
  peak_memory_max_gb: null

gate_policy:
  enforce_on_v1_100_images: false
  report_confidence_interval: true
  confidence_interval_method: wilson
  confidence_level: 0.95
  minimum_samples_for_preliminary_acceptance: 1000
  minimum_samples_for_high_confidence_failure_rate: 3000

ranking_policy:
  raw_metrics_primary: true
  gate_before_composite_score: true
  score_weights_configurable: true
```

Một model vi phạm production gate không thể trở thành recommendation nhờ composite score cao.

### Q40. Định dạng output bắt buộc?

- [x] A. JSONL raw predictions + JSON summary + CSV + offline HTML (khuyến nghị)
- [ ] B. Chỉ JSON/CSV
- [ ] C. Thêm Parquet/database: ____________________

### Q41. Thời gian lưu raw predictions và ảnh lỗi?

- [x] A. Lưu toàn bộ cho đến khi xóa thủ công (phù hợp dataset 100–10.000 ảnh)
- [ ] B. Chỉ lưu summary và top lỗi
- [ ] C. Retention: ____________________ ngày

### Q42. Cost metric có cần trong V1 không?

- [x] A. Có nếu nhập hardware/electricity assumptions; luôn gắn nhãn estimated (khuyến nghị)
- [ ] B. Chưa cần
- [ ] C. Có dữ liệu cost cụ thể: ____________________

Nếu không cung cấp giá phần cứng, vòng đời kỳ vọng hoặc giá điện thì cost metrics phải trả N/A; không dùng giá trị ngầm định.

### Q43. Một run bị lỗi giữa chừng có cần resume không?

- [x] A. Có checkpoint theo model/image và resume được (khuyến nghị)
- [ ] B. Không; chạy lại toàn bộ
- [ ] C. Chỉ resume theo model

### Q44. Mức test/CI cần thiết?

- [x] A. Unit test + mock integration test trong CI; model smoke test thủ công trên GX10 (khuyến nghị)
- [ ] B. CI phải chạy model thật
- [ ] C. Chỉ unit test

### Q45. Definition of Done có giữ nguyên như đặc tả không?

- [x] A. Có: run_all validate → load → warmup → benchmark → metrics → unload → reports → leaderboard
- [ ] B. Điều chỉnh thành: __________________________________________

### Q46. Những điều gì là explicit non-goal của V1? (có thể chọn nhiều)

- [x] A. Web frontend tương tác
- [x] B. Multi-user scheduler
- [x] C. Cloud deployment
- [x] D. Model fine-tuning
- [x] E. ERP integration thật
- [ ] F. Khác: ____________________

---

## 8. Rủi ro cần xác nhận

Đánh dấu các rủi ro chấp nhận được:

- [x] Một số model Tier A có thể DEPENDENCY_ERROR hoặc NOT_SUPPORTED trên ARM64/GB10.
- [x] Một số framework có thể không cung cấp NVML/VRAM metric giống GPU rời.
- [x] VLM end-to-end có thể trả output không đồng nhất; adapter phải chuẩn hóa hoặc ghi INVALID_OUTPUT.
- [x] License/model card có thể thay đổi; kết quả phải gắn với ngày và exact revision.
- [x] Accuracy ranking có thể không ổn định nếu dataset quá nhỏ hoặc lệch điều kiện ảnh.
- [x] full_label_exact_match sẽ thấp hơn field accuracy khi chỉ một field sai.
- [ ] Khác: ______________________________________________________

## Understanding Lock

**Status:** CONFIRMED — architecture accepted after D-015 to D-018

### Understanding Summary

- Xây một OCR Model Evaluation Lab production-grade để chọn model và cấu hình deployment dựa trên customer holdout, không dựa vào benchmark marketing.
- Người dùng V1 là ML/MLOps và Backend/Platform engineer; vận hành offline bằng CLI, một người trên một ASUS Ascent GX10, chạy từng model có isolation.
- Delivery đầu hỗ trợ Tier A: PP-OCRv6, GLM-OCR, PaddleOCR-VL và MonkeyOCRv2; adapter lỗi vẫn tạo status có nguyên nhân và không làm dừng suite.
- Dataset ban đầu khoảng 100 ảnh, mở rộng đến 10.000+, mỗi ảnh đúng một tem vật lý, dynamic schema theo label_type/sample và barcode được benchmark riêng.
- Exact field match và full-label exact match là metric quyết định; OCR/text quality và end-to-end extraction được report riêng cùng latency, throughput, reliability và resource usage.
- 100 ảnh chỉ dùng exploratory; production gates/weights configurable, có confidence interval, và recommendation chắc chắn chỉ áp dụng cho phần cứng đã benchmark.
- V1 không gồm web frontend, multi-user scheduler, cloud deployment, model fine-tuning, ERP integration thật, multi-label evaluation hoặc fallback execution.

### Assumptions

- Official repository/model card thắng mọi assumption trong tài liệu; exact revision, backend và license phải được xác minh trước khi implement adapter.
- Dataset holdout không được dùng để tune prompt, regex hoặc extraction config; mọi config được freeze và version hóa trước run chính thức.
- Không có bbox ground truth trong V1 nên detection metrics là N/A; schema giữ khả năng bổ sung bbox/multi-label ở phase sau.
- Thiếu confidence, cost inputs hoặc hardware counter thì metric tương ứng là N/A, không được fake hoặc ước lượng ngầm.
- Tier B/C chỉ được rank khi có result hợp lệ và được tách leaderboard theo tier.
- Chưa có deadline cố định; scope và correctness được ưu tiên hơn thời gian.

### Open Questions

Không còn open question nào chặn việc chuyển sang phân tích kiến trúc. Model compatibility và resource-counter availability trên ARM64/GB10 là các mục cần xác minh bằng tài liệu chính thức và smoke test, không phải giả định thiết kế.
