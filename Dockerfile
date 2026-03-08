# slim python
FROM python:3.11-slim

WORKDIR /app

# to make use of the cache we copy the requirements file first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the source code
COPY . .

# port 8000 for fastAPI
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]