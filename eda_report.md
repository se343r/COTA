# Phân tích Tập dữ liệu OCR Công tơ điện

Báo cáo tổng hợp số liệu thống kê sau quá trình gán nhãn của người dùng.

## 1. Trạng thái lọc ảnh (Filtering)
- **Tổng số ảnh đã lọc:** 1089
- `keep`: 619 ảnh (56.8%)
- `reject-dup`: 298 ảnh (27.4%)
- `reject-blur`: 72 ảnh (6.6%)
- `reject-other`: 40 ảnh (3.7%)
- `reject-dark`: 26 ảnh (2.4%)
- `reject-partial`: 17 ảnh (1.6%)
- `reject-nometer`: 17 ảnh (1.6%)

## 2. Phân loại Công tơ (Meter Types)
- **Cơ (Từng số):** 428 ảnh
- **Điện tử (Chuỗi số):** 191 ảnh

## 3. Chỉ số toàn trình (OCR Sequences)
- **Tổng số ảnh đã gán chuỗi:** 618

**Phân bố độ dài chuỗi:**
| Độ dài | Số lượng |
|---|---|
| 3 ký tự | 1 |
| 4 ký tự | 2 |
| 5 ký tự | 16 |
| 6 ký tự | 22 |
| 7 ký tự | 407 |
| 8 ký tự | 161 |
| 9 ký tự | 7 |
| 10 ký tự | 2 |

## 4. Chi tiết Bounding Boxes (Crops)
- **Số lượng ảnh có lưu BBox:** 618

**Phân bố số lượng box trên mỗi ảnh:**
| Số BBox | Số lượng ảnh |
|---|---|
| 0 box | 97 |
| 1 box | 94 |
| 5 box | 2 |
| 6 box | 425 |

**Thuộc tính đặc biệt của BBox:**
- `is_between` (Số bị cuộn nửa): 143
- `is_decimal` (Dấu thập phân): 424
- `unreadable` (Không thể đọc): 48

**Tần suất xuất hiện các chữ số:**
| Ký tự | Số lần xuất hiện |
|---|---|
| `1` | 398 |
| `0` | 357 |
| `2` | 321 |
| `4` | 231 |
| `3` | 220 |
| `5` | 210 |
| `7` | 202 |
| `6` | 200 |
| `8` | 196 |
| `9` | 177 |
| `unreadable` | 48 |