.PHONY: setup clean update

VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

setup: $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

$(VENV):
	python3 -m venv $(VENV)

clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

update: $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install --upgrade -r requirements.txt

.env:
	@echo "Creating .env file..."
	@echo "OPENAI_API_KEY=" > .env
	@echo "TAVILY_API_KEY=" >> .env
	@echo "LANGSMITH_API_KEY=" >> .env

# launch langgraph studio
run_studio:
	langgraph dev

