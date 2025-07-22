install:
	uv venv
	uv pip install gunicorn
run:
	uv run gunicorn -w 4 -b 0.0.0.0:3000 app:app
