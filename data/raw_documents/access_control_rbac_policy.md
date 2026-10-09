# Chính Sách Kiểm Soát Truy Cập và Phân Quyền Theo Vai Trò (RBAC Policy)

## 1. Nguyên Tắc Cốt Lõi
Hệ thống quản trị tài nguyên thông tin áp dụng triệt để hai nguyên tắc:
* **Nguyên tắc Đặc quyền Tối thiểu (Principle of Least Privilege - PoLP):** Người dùng chỉ được cấp quyền tối thiểu cần thiết để hoàn thành công việc.
* **Nguyên tắc Phân tách Trách nhiệm (Separation of Duties - SoD):** Các tác vụ nhạy cảm (như phê duyệt giải ngân, cấp chứng chỉ bảo mật) bắt buộc phải qua 2 cấp xác thực độc lập.

## 2. Hệ Thống Phân Quyền Vai Trò (Role Hierarchy)
1. **Viewer (Người xem):** Chỉ đọc các tài liệu hướng dẫn và chính sách chung. Không có quyền tải xuống tài liệu bảo mật.
2. **Contributor (Cộng tác viên):** Đọc và đóng góp tài liệu vào thư viện dự án được chỉ định.
3. **Data Custodian (Người giám sát dữ liệu):** Phê duyệt quyền truy cập dữ liệu nhạy cảm, thiết lập nhãn phân loại dữ liệu (Public, Internal, Confidential, Restricted).
4. **Security Administrator (Quản trị bảo mật):** Quản lý chứng chỉ khóa, cấu hình xác thực MFA và rà soát audit log.

## 3. Quản Lý Khóa và Phiên Làm Việc
* Mọi khóa API và token truy cập phải tuân theo chu kỳ luân chuyển tối đa 90 ngày.
* Phiên làm việc không hoạt động (idle session) sẽ tự động bị ngắt kết nối sau 15 phút.
