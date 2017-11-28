#.PHONY build run

DATA_DIR ?= /tmp

build:
	docker build --rm -t facebook-lm .

run:
	docker run --rm -it -v $(DATA_DIR):/data -v $(CURDIR):/code facebook-lm bash
