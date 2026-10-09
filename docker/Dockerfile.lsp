# The editor's language server: basedpyright behind a WebSocket bridge (nanoscope.server.lsp_bridge).
# It starts from the nanoscope image so the server sees the same torch and nanoscope the code runs
# against (hovers show torch's signatures); basedpyright's wheel brings its own Node.
#   docker build -f docker/Dockerfile.lsp -t nanoscope:lsp .
ARG BASE=ghcr.io/almajd3713/nanoscope:cpu
FROM ${BASE}
USER root
RUN pip install --no-cache-dir basedpyright==1.40.2
USER 1000
# the workspace is mounted read only; the bridge starts a server per connection, in that folder
ENV NANOSCOPE_LSP_ROOT=/nanoscope/workspace
EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:3000/health', timeout=4)"
CMD ["python", "-m", "uvicorn", "nanoscope.server.lsp_bridge:app", "--host", "0.0.0.0", "--port", "3000"]
