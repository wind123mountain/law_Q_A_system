FROM python:3.11-slim

# Cài đặt thư viện hệ thống cần thiết
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     gcc \
#     && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Sao chép requirement và cài đặt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ project vào container
COPY . .

# Cấu hình lệnh khởi chạy
CMD ["tail", "-f", "/dev/null"]
