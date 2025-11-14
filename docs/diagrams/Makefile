.PHONY: all clean

all: diagrams

diagrams:
	@echo "Building network diagrams..."
	@mkdir -p ../images
	cabal run network-diagrams
	@echo "Done! Diagrams saved to ../images/"

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf dist-newstyle
	@rm -f ../images/*.svg
	@echo "Clean complete."
