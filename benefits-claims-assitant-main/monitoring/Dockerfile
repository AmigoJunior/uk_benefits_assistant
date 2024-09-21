FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pipenv install --no-cache-dir -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py"]