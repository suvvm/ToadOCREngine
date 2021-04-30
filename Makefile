# info

# help: Help for this project
help: Makefile
	@echo "Usage:\n  make [command]"
	@echo
	@echo "Available Commands:"
	@sed -n 's/^##//p' $< | column -t -s ':' |  sed -e 's/^/ /'

## build: Compile the binary. Copy binary product to current directory
build:
	@sh build.sh

## run: Build and run, run command `train cnn` by default
run: build
	@sh output/bootstrap.sh

## clean: Clean output
clean:
	rm -rf output
	rm -f toad_ocr_engine

## generate: generate idl code
generate:
	@sh idl_generate.sh