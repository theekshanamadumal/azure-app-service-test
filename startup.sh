python -m server
# python -m uvicorn server:app --host 0.0.0.0
# gunicorn -w 4 -k uvicorn.workers.UvicornWorker server:app 