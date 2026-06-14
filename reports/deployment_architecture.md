# Deployment Architecture

The system uses a containerized microservices architecture to ensure reproducibility across environments.

## Services
1. **Frontend**: A React + TypeScript SPA served via Vite (`node:18-alpine`). Communicates via HTTP REST.
2. **Backend (Model Server)**: A FastAPI application running on `python:3.11-slim`. Wraps the PyTorch/PyG inference logic.
3. **Registry Mount**: Docker bind-mounts `./models` directly into the backend container, allowing hot-swapping without container restarts.

## Scaling
The backend relies on the `uvicorn` ASGI server. By adjusting the number of workers in the `docker-compose.yml`, the API can horizontally scale to handle parallel neighborhood analyses.
