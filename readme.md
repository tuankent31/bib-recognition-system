# Cách sử dụng dự án
## Bước 1: Giải nén file dự án
Giải nén file `bib_recognition_sys.zip`

## Bước 2: Tải file ảnh để thử nghiệm
- Tải file ảnh từ link: [https://drive.google.com/file/d/1xX26Vhrx_b0sQuAbL2k8G5ha7yw05BHV/view?usp=sharing](https://drive.google.com/file/d/1bYXd7yzxdwzYr3q0p1ELTKpA-ZxmLXjK/view)
- Giải nén file và di chuyển ảnh vào thư mục `static/images` 

## Bước 3: Cài đặt thư viện và mô hình
- Tải các mô hình và chuyển vào thư mục `models` bằng link: https://drive.google.com/file/d/14j0L_FluRoc0HgkE_xhzzQQDLZ1K39oH/view?usp=sharing
- Mở cửa sổ terminal và tải các thư viện và module cần thiết bằng lệnh `pip install -r requirements.txt`

## Bước 4: Cài đặt cơ sở dữ liệu
- Tạo cơ sở dữ liệu trong SQL Server tên là `bibdata`
- Chỉnh sửa `SQL_SERVER, SQL_USER, SQL_PASSWORD` trong file `config.py` tương ứng với máy tính.
- Chạy file `create_data.py` trên terminal bằng lệnh python `python create_data.py` để tạo dữ liệu ảo

## Bước 5: Chạy dự án
- Trên terminal nhập: `python app.py` để chạy hệ thống.
- Đăng nhập với thông tin:
`username: admin`
`password: password123`

DEMO SẢN PHẨM: https://bit.ly/3XHKkEB
