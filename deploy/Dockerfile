FROM vllm/vllm-openai:v0.31.0
WORKDIR /app
COPY requirements-serving.txt /app/
RUN python3 -m pip install --no-cache-dir -r requirements-serving.txt
COPY baseline /app/baseline
COPY submission /app/submission
COPY configs /app/configs
ENTRYPOINT ["python3", "-m", "submission.launch"]
CMD ["--config", "/deployment/serving.json", "--api-host", "0.0.0.0"]
