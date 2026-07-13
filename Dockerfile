FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN sed -i '/torch/d' requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 8000

CMD ["python","-m","uvicorn","app:app","--host","0.0.0.0","--port","8000"]