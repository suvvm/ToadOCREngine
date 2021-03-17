# info

RUN_NAME="toad_ocr_engine"

# help: Help for this project
help: Makefile
	@echo "Usage:\n  make [command]"
	@echo
	@echo "Available Commands:"
	@sed -n 's/^##//p' $< | column -t -s ':' |  sed -e 's/^/ /'

## build: Compile the binary.
build:
	@sh build.sh

## run: Build and run
run:
	$(MAKE) build
	./output/bin/$(RUN_NAME)

## clean: Clean output
clean:
	rm -rf output
