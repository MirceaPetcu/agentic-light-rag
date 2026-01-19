# Alternative Dockerfile if you need custom configuration
FROM vllm/vllm-openai:latest

# Install additional dependencies if needed
# RUN pip install <additional-packages>

# Set working directory
WORKDIR /app

# Environment variables
ENV MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
ENV HOST="0.0.0.0"
ENV PORT="8000"

# Expose port
EXPOSE 8000

# Default command (can be overridden in docker-compose.yml)
CMD python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --host $HOST \
    --port $PORT
