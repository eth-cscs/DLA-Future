DLAF_SOURCE_DIR=$(shell pwd)/..

doc:
	sed 's|$${DLAF_SOURCE_DIR}|$(DLAF_SOURCE_DIR)|g' Doxyfile | doxygen -
