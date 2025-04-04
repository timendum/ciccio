set dotenv-load

default:
  just --list

download FOLDER="o" *FLAGS:
  uv run main.py {{FLAGS}} download {{FOLDER}}

train *FLAGS:
  uv run main.py {{FLAGS}} train

split *FLAGS:
  uv run main.py {{FLAGS}} split

setup:
  uv sync

lint:
  uvx ruff check .

cformat:
  uvx ruff format --check .

format:
  uvx ruff format .