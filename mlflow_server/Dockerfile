FROM python:3.14-slim
RUN pip install --no-cache-dir mlflow==3.17.0
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", \
     "--allowed-hosts", "mlflow:5000,localhost:5000,127.0.0.1:5000", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", "--serve-artifacts"]