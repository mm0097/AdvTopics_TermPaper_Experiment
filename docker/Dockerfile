FROM python:3.10-slim-bullseye
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential     gcc     && pip install --no-cache-dir     jupyter_client==8.6.1     ipykernel     numpy     pandas     matplotlib     seaborn     scipy     scikit-learn     && apt-get purge -y --auto-remove build-essential gcc     && rm -rf /var/lib/apt/lists/*
CMD ["jupyter", "kernel", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]