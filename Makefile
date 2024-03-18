format:
	@poetry run autoflake -i -r .
	@poetry run black .
	@poetry run isort .
