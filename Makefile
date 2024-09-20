initial:
	pip install pre-commit
	pip install -r requirements.txt --no-cache-dir
	pre-commit install

run_tests: initial
	PYTHONPATH=. pytest tests/tests.py

build_server:
	 docker build -t service .

run_server_docker: run_tests
	docker run -d --name mycontainer -p 5053:5053 service

run_server_local: run_tests
	python app.py

stop_server:
	docker rm $(docker stop $(docker ps -a -q --filter ancestor=service --format="{{.ID}}"))
