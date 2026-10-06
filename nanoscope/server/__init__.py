"""The HTTP API: FastAPI over the library, started with `nanoscope serve`.

Needs the `server` extra (`pip install "nanoscope-lab[server]"`). The API process reads files and
the job queue and answers; it never imports or runs user code (plan 8.1): anything that does
goes to a worker as a job.
"""
