# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** [Tên sinh viên]
**Nhóm:** [Tên nhóm]
**Ngày:** [Ngày nộp]

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> *Viết 1-2 câu:*

**Ví dụ HIGH similarity:**
- Sentence A:
- Sentence B:
- Tại sao tương đồng:

**Ví dụ LOW similarity:**
- Sentence A:
- Sentence B:
- Tại sao khác:

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> *Viết 1-2 câu:*

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> *Đáp án:*

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> *Viết 1-2 câu:*

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Customer support knowledge base / help-center documentation

**Tại sao nhóm chọn domain này?**
> Nhóm chọn domain customer support vì tài liệu dạng FAQ và troubleshooting rất dễ kiếm, có cấu trúc rõ ràng, và phù hợp với retrieval hơn nhiều so với văn bản tự do. Bộ tài liệu này cũng đủ đa dạng để thử các câu hỏi về account, password, billing, refund, rate limit, và cả tài liệu nội bộ cần metadata filtering. Ngoài ra, domain này gần với bài toán RAG thực tế: tìm đúng hướng dẫn và tránh lấy nhầm tài liệu internal cho người dùng cuối.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | `account_email_change.md` | OpenAI Help Center | 1999 | `doc_id`, `title`, `category=account`, `audience=customer`, `language=en`, `source`, `last_updated`, `sensitivity=public` |
| 2 | `password_reset_help.md` | OpenAI Help Center | 1815 | `doc_id`, `title`, `category=password`, `audience=customer`, `language=en`, `source`, `last_updated`, `sensitivity=public` |
| 3 | `billing_renewal_failure.md` | OpenAI Help Center | 1789 | `doc_id`, `title`, `category=billing`, `audience=customer`, `language=en`, `source`, `last_updated`, `sensitivity=public` |
| 4 | `refund_request_guide.md` | OpenAI Help Center | 1952 | `doc_id`, `title`, `category=refund`, `audience=customer`, `language=en`, `source`, `last_updated`, `sensitivity=public` |
| 5 | `service_limit_429.md` | OpenAI Help Center | 1867 | `doc_id`, `title`, `category=service_limit`, `audience=customer`, `language=en`, `source`, `last_updated`, `sensitivity=public` |
| 6 | `internal_escalation_playbook.md` | Internal support / handbook reference | 1944 | `doc_id`, `title`, `category=escalation`, `audience=internal_support`, `language=en`, `source`, `last_updated`, `sensitivity=internal` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `doc_id` | string | `kb_refund_001` | Định danh duy nhất cho mỗi tài liệu, hữu ích khi quản lý, debug hoặc xóa tài liệu khỏi vector store |
| `title` | string | `How to Request a Refund for a ChatGPT Subscription` | Giúp nhận diện nhanh nội dung tài liệu và trình bày nguồn rõ ràng trong kết quả retrieval |
| `category` | string | `account`, `password`, `billing`, `refund`, `service_limit`, `escalation` | Cho phép lọc theo chủ đề để tăng precision, nhất là khi query thuộc một mảng hỗ trợ cụ thể |
| `audience` | string | `customer`, `internal_support` | Rất quan trọng để tránh trả tài liệu nội bộ cho người dùng cuối và hỗ trợ metadata filtering |
| `language` | string | `en` | Hữu ích khi sau này mở rộng sang tài liệu đa ngôn ngữ hoặc cần giới hạn theo ngôn ngữ người hỏi |
| `source` | string | `https://help.openai.com/en/articles/...` | Giúp truy vết nguồn gốc tài liệu và kiểm tra độ tin cậy của câu trả lời |
| `last_updated` | string | `2026-04-10` | Hữu ích nếu sau này cần ưu tiên tài liệu mới hơn hoặc theo dõi độ tươi của dữ liệu |
| `sensitivity` | string | `public`, `internal` | Hỗ trợ kiểm soát truy cập và giảm nguy cơ retrieve nhầm tài liệu nhạy cảm |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| `account_email_change.md` | FixedSizeChunker (`fixed_size`) | 6 | 374.8 | Trung bình, độ dài ổn định nhưng có thể cắt giữa một chuỗi hướng dẫn |
| `account_email_change.md` | SentenceChunker (`by_sentences`) | 11 | 180.2 | Tốt, dễ đọc nhưng bị chia khá nhỏ |
| `account_email_change.md` | RecursiveChunker (`recursive`) | 7 | 283.9 | Tốt, giữ được các cụm ý theo đoạn và không quá vụn |
| `refund_request_guide.md` | FixedSizeChunker (`fixed_size`) | 6 | 367.0 | Trung bình, có nguy cơ cắt giữa policy và timeline |
| `refund_request_guide.md` | SentenceChunker (`by_sentences`) | 7 | 277.0 | Tốt, hợp với tài liệu dạng hướng dẫn theo bước |
| `refund_request_guide.md` | RecursiveChunker (`recursive`) | 6 | 323.7 | Tốt, cân bằng giữa độ dài chunk và ý trọn vẹn |
| `internal_escalation_playbook.md` | FixedSizeChunker (`fixed_size`) | 6 | 365.7 | Trung bình, dễ cắt ngang bullet hoặc checklist nội bộ |
| `internal_escalation_playbook.md` | SentenceChunker (`by_sentences`) | 5 | 387.2 | Khá tốt nhưng một số chunk hơi dài |
| `internal_escalation_playbook.md` | RecursiveChunker (`recursive`) | 7 | 276.1 | Tốt nhất, giữ cấu trúc playbook và nhóm ý rõ hơn |

### Strategy Của Tôi

**Loại:** `RecursiveChunker`

**Mô tả cách hoạt động:**
> `RecursiveChunker` chia tài liệu theo thứ tự ưu tiên từ separator lớn đến nhỏ như `\n\n`, `\n`, `. `, khoảng trắng và cuối cùng mới fallback sang cắt cứng nếu cần. Cách này giúp thuật toán ưu tiên giữ các ranh giới tự nhiên như đoạn văn, section hoặc câu trước khi buộc phải tách nhỏ hơn. Khi một đoạn vẫn vượt quá `chunk_size`, nó tiếp tục đệ quy với separator nhỏ hơn để tạo chunk hợp lệ. Nhờ đó chunk vừa đủ ngắn để retrieve, nhưng vẫn giữ được ý nghĩa khá trọn vẹn.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Bộ tài liệu của nhóm là customer support knowledge base, gồm các bài help-center theo bước và một playbook nội bộ có nhiều section, bullet, điều kiện và ngoại lệ. `RecursiveChunker` tận dụng tốt cấu trúc này hơn `FixedSizeChunker` vì không cắt cơ học theo số ký tự, đồng thời ổn định hơn `SentenceChunker` khi gặp đoạn dài hoặc nhiều bullet. Với domain này, giữ nguyên từng cụm hướng dẫn hoặc từng nhóm điều kiện rất quan trọng để retrieval trả đúng evidence.


### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| 3 tài liệu baseline | best baseline: `SentenceChunker` | 23 | 281.5 | Dễ đọc và bám ranh giới câu, nhưng đôi khi chia quá nhỏ hoặc tạo chunk hơi dài ở playbook nội bộ |
| 3 tài liệu baseline | **của tôi: `RecursiveChunker`** | 20 | 294.6 | Cân bằng tốt hơn giữa độ dài chunk và context; phù hợp hơn cho tài liệu có paragraph, steps và internal notes |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | RecursiveChunker (chunk_size=400) | 6/10 (proxy nội bộ) | Giữ context tốt, phù hợp với tài liệu có section, steps và internal notes | Tạo nhiều chunk hơn và chưa vượt trội rõ rệt về điểm số khi chỉ dùng `_mock_embed` |
| Trần Sỹ Minh Quân| RecursiveChunker (chunk_size=420) | 7/10 | Giữ được ngữ cảnh theo section và bullet khá tốt, hợp tài liệu support có cấu trúc rõ ràng | Nếu query quá mơ hồ hoặc quá ngắn thì đôi lúc chunk top-1 vẫn lệch sang tài liệu gần nghĩa hơn |
| Ngô Quang Tăng | SentenceChunker (3 sent/chunk) | 8.9 | Giữ instructions trọn vẹn, semantic units | Chunks không đều (refund: 277 chars, escalation: 387 chars) |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Dựa trên kết quả so sánh hiện tại trong nhóm, `SentenceChunker (3 sent/chunk)` là strategy tốt nhất cho domain này vì có `Retrieval Score` cao nhất (`8.9`) và giữ được các hướng dẫn theo từng semantic unit khá trọn vẹn. Điều này đặc biệt phù hợp với bộ tài liệu customer support của nhóm, vì phần lớn nội dung được viết theo dạng step-by-step instructions, policy notes và troubleshooting guidance. Tuy nhiên, `RecursiveChunker` vẫn là một lựa chọn rất mạnh khi tài liệu có nhiều section, bullet và internal notes, nên nếu mở rộng sang tài liệu dài và cấu trúc phức tạp hơn thì nó vẫn đáng cân nhắc.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Tôi dùng regex `(?<=[.!?])(?:\s+|\n+)` để tách câu theo dấu kết câu kết hợp khoảng trắng hoặc xuống dòng. Sau khi split, tôi `strip()` từng câu và loại bỏ phần rỗng để tránh sinh chunk lỗi khi văn bản có nhiều khoảng trắng hoặc dòng trống. Cuối cùng, các câu được gom lại theo `max_sentences_per_chunk`, nên cách làm này đơn giản nhưng đủ ổn định cho bộ tài liệu của lab.

**`RecursiveChunker.chunk` / `_split`** — approach:
> `chunk()` chỉ kiểm tra input rỗng rồi chuyển sang `_split()` để xử lý chính. Trong `_split()`, nếu đoạn hiện tại đã ngắn hơn hoặc bằng `chunk_size` thì trả về ngay; nếu chưa, thuật toán thử từng separator theo thứ tự ưu tiên (`\n\n`, `\n`, `. `, khoảng trắng, rồi cắt cứng) để chia đoạn một cách tự nhiên nhất. Base case là khi đoạn đã đủ ngắn hoặc khi không còn separator nào để thử, lúc đó hàm fallback về cắt text theo `chunk_size`.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Tôi chuẩn hóa mỗi document thành một record gồm `id`, `content`, `metadata` và `embedding`, sau đó lưu vào `self._store`; nếu ChromaDB khả dụng thì cũng thêm record vào collection tương ứng. `add_documents()` gọi embedding function cho từng document rồi lưu toàn bộ record. `search()` embed câu query, tính điểm bằng dot product giữa vector query và vector của từng record, rồi sort giảm dần theo `score`.

**`search_with_filter` + `delete_document`** — approach:
> Tôi lọc theo metadata trước rồi mới chạy similarity search, vì nếu search trên toàn bộ store rồi mới filter thì top-k có thể sai hoặc bỏ lỡ kết quả tốt trong tập con cần tìm. `search_with_filter()` giữ lại các record có metadata khớp với `metadata_filter`, sau đó gọi lại `_search_records()` trên tập đã lọc. `delete_document()` xóa toàn bộ record có `metadata["doc_id"]` trùng với document cần xóa và, nếu có ChromaDB, cũng xóa các `id` tương ứng trong collection.

### KnowledgeBaseAgent

**`answer`** — approach:
> `KnowledgeBaseAgent.answer()` trước hết retrieve `top_k` chunk liên quan nhất từ `EmbeddingStore`. Sau đó tôi ghép các chunk này thành một khối `Context`, trong đó mỗi chunk có số thứ tự, `score` và `doc_id` để dễ truy vết nguồn. Prompt cuối cùng được tổ chức theo dạng “chỉ trả lời dựa trên context; nếu context không đủ thì nói rõ”, rồi truyền sang `llm_fn`.

### Test Results

```
============================= test session starts =============================
collected 42 items

======================== 42 passed, 1 warning in 0.09s ========================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | A customer can change their email from Settings > Account on ChatGPT Web. | Users can update their account email in ChatGPT settings on the web. | high | 0.137 | Có |
| 2 | If you do not receive the password reset email, check spam and verify the signup inbox. | Customers should check spam or junk folders when reset emails do not arrive. | high | -0.270 | Không |
| 3 | A 429 error can be handled with exponential backoff and retry delays. | Refund requests for Apple subscriptions must be handled directly with Apple. | low | 0.004 | Có |
| 4 | Renewal payment failures may be caused by bank-side security checks or incorrect billing details. | Billing renewals can fail because of bank checks, expired cards, or wrong payment information. | high | -0.003 | Không |
| 5 | Internal escalation should be considered if an emergency lasts more than 3 hours. | You can request a refund through the Help Center chat widget. | low | 0.006 | Có |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Kết quả bất ngờ nhất là Pair 2 và Pair 4, vì về mặt nghĩa hai cặp này khá giống nhau nhưng actual score lại âm hoặc gần bằng 0. Điều này cho thấy trong bài lab hiện tại, `_mock_embed` phù hợp để test logic hệ thống hơn là đo semantic similarity thật sự. Nếu dùng embedding model thật, các cặp paraphrase như vậy nhiều khả năng sẽ có score cao và sát trực giác hơn.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | How can a customer change the email address on their OpenAI account? | A customer can change their email from `Settings > Account` on ChatGPT Web if the account supports email management. This is not supported for phone-number-only accounts, Enterprise SSO accounts, or some enterprise-verified personal accounts. After the change, the user is signed out and must log in again with the new email. |
| 2 | What should a customer do if they do not receive the password reset email? | The customer should check the spam/junk folder, confirm they are checking the same inbox used during signup, and verify there is no typo in the email address. If the account was created only with Google, Apple, or Microsoft login, password recovery must be done through that provider instead. |
| 3 | What are the recommended steps when a ChatGPT Plus or Pro renewal payment fails? | The customer should clear browser cache and cookies, contact the bank to check for blocks or security flags, verify billing and card details, and confirm the country or region is supported. If the payment still fails, they should contact support through the Help Center chat widget. |
| 4 | How should a customer handle a 429 Too Many Requests error? | A 429 error means the organization exceeded its request or token rate limit. The recommended solution is exponential backoff: wait, retry, and increase the delay after repeated failures. The customer should also reduce bursts, optimize token usage, and consider increasing the usage tier if needed. |
| 5 | When should an active customer emergency be escalated, and who should be contacted first? | Escalation should be considered when the emergency lasts more than 3 hours without clear resolution, involves multiple simultaneous customer issues, blocks critical outside work, or requires broader coordination. A Support Manager On-call should be consulted, and the account CSM should usually be contacted first as the escalation DRI. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | How can a customer change the email address on their OpenAI account? | `internal_escalation_playbook.md` - top-1 bị lệch sang tài liệu internal escalation, không chứa hướng dẫn đổi email | 0.158 | Không | Agent answer bị nhiễu, trộn nội dung 429 / password reset / email change nên không trả lời đúng gold answer |
| 2 | What should a customer do if they do not receive the password reset email? | `account_email_change.md` - top-1 nói về đổi email, không phải reset password | 0.223 | Không | Agent answer bị kéo sang escalation và account management nên chưa bám đúng câu hỏi reset email |
| 3 | What are the recommended steps when a ChatGPT Plus or Pro renewal payment fails? | `password_reset_help.md` - top-1 không đúng chủ đề billing renewal failure | 0.225 | Không | Agent answer chủ yếu nói về 429, refund và password reset nên không grounded đúng vào gold answer về failed renewal |
| 4 | How should a customer handle a 429 Too Many Requests error? | `password_reset_help.md` - top-1 lệch, nhưng `service_limit_429.md` có xuất hiện trong top-3 và agent vẫn dùng được nội dung đúng | 0.127 | Có | Agent answer nêu được ý chính về 429, rate limit, exponential backoff và giảm burst |
| 5 | When should an active customer emergency be escalated, and who should be contacted first? | `refund_request_guide.md` - top-1 sai chủ đề, nhưng `internal_escalation_playbook.md` có trong top-3 | 0.230 | Có | Agent answer vẫn trích được nội dung đúng từ playbook: cân nhắc escalation sau 3 giờ và nên liên hệ CSM / Support Manager On-call |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 2 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Điều tôi học được nhiều nhất từ thành viên khác trong nhóm là `SentenceChunker` có thể hoạt động rất tốt trên bộ tài liệu customer support khi các bài viết được tổ chức theo từng bước và từng semantic unit rõ ràng. So với suy nghĩ ban đầu của tôi rằng `RecursiveChunker` sẽ luôn mạnh hơn, kết quả nhóm cho thấy strategy đơn giản hơn vẫn có thể thắng nếu nó khớp đúng với cấu trúc dữ liệu. Điều này nhắc tôi rằng lựa chọn chunking nên dựa trên đặc điểm tài liệu thực tế, không nên chỉ dựa trên độ phức tạp của thuật toán.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Qua phần demo, bài học đáng chú ý nhất là retrieval quality không phụ thuộc riêng vào code search mà phụ thuộc rất mạnh vào cách chuẩn bị dữ liệu, đặc biệt là chunk boundaries và metadata. Một hệ thống có implementation đúng vẫn có thể cho kết quả yếu nếu tài liệu bị chunk quá vụn, thiếu metadata hoặc benchmark queries không đủ sát với nội dung tài liệu. Vì vậy, phần data strategy thực tế quan trọng không kém phần code.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Nếu làm lại, tôi sẽ index theo chunk thay vì lưu nguyên toàn bộ document như trong bản demo đơn giản hiện tại, vì như vậy retrieval sẽ bám sát từng đoạn evidence hơn. Tôi cũng sẽ dùng metadata filtering rõ hơn cho các trường như `audience`, `category` và `sensitivity` để tránh kéo nhầm tài liệu internal vào các query của customer. Ngoài ra, tôi sẽ chuẩn hóa benchmark queries tốt hơn để có ít nhất một vài câu hỏi cho từng category chính như account, billing, rate limit và escalation.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 14 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 6 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** | | **84 / 100** |
