cc = gcc
cfalgs = -fPIC -shared

.PHONY: clear

cfunctions/linalg_functions.so: cfunctions/linalg_functions.c
	$(cc) $(cfalgs) -o $@ $^

clear:
	rm cfunctions/linalg_functions.so