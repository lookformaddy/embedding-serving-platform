FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY examples/ ./examples/
ENV PYTHONPATH=/app/src
CMD ["python", "examples/run_demo.py"]
