# Sổ Tay Quy Trình Ứng Phó Sự Cố An Ninh Thông Tin (Incident Response Playbook)

## 1. Mục đích và Phạm vi
Tài liệu quy định quy trình chuẩn SOP khi phát hiện sự cố an ninh mạng, rò rỉ dữ liệu hoặc xâm nhập trái phép vào hạ tầng đám mây nội bộ.

## 2. Các Mức Độ Nghiêm Trọng (Severity Levels)
* **Severity 1 (Critical):** Rò rỉ dữ liệu khách hàng, mã nguồn chính hoặc mất kiểm soát tài khoản quản trị viên root/global admin. Yêu cầu phong tỏa hệ thống trong vòng 15 phút.
* **Severity 2 (High):** Tấn công từ chối dịch vụ DoS diện rộng, phát hiện mã độc ransomware trên máy trạm nội bộ. Thời gian cô lập tối đa 1 giờ.
* **Severity 3 (Medium/Low):** Cảnh báo đăng nhập bất thường, nghi ngờ lừa đảo qua email (phishing) chưa gây hậu quả.

## 3. Quy Trình 4 Bước Xử Lý
1. **Phát hiện & Báo cáo:** Báo cáo ngay cho Đội Phản Ứng Nhanh (CSIRT) qua cổng an ninh nội bộ hoặc hotline bảo mật.
2. **Cách ly & Ngăn chặn:** Lập tức ngắt kết nối mạng của thiết bị bị nhiễm, vô hiệu hóa phiên làm việc (revoke session tokens), và đổi mật khẩu quản trị.
3. **Điều tra & Khắc phục:** Thu thập nhật ký truy cập (audit logs), phục hồi hệ thống từ bản sao lưu sạch (clean backup).
4. **Báo cáo Hậu Sự Cố (Post-Mortem):** Soạn thảo báo cáo phân tích nguyên nhân gốc rễ (Root Cause Analysis - RCA) trong vòng 48 giờ.
