CFLAGS ?= -std=c89 -Wall -Wextra -pedantic
CXXFLAGS ?= -std=c++98 -Wall -Wextra -pedantic
CLANG_TIDY ?= clang-tidy

HAS_CLANG_TIDY := $(shell command -v $(CLANG_TIDY) 2> /dev/null)

all: test lint

lint:
ifdef HAS_CLANG_TIDY
	$(CLANG_TIDY) -checks='*' nn_test.c
endif
	echo '#include "nn.h"' > nn_lint.c
	echo '#include "nn.h"' > nn_lint.cxx
	$(CC) $(CFLAGS) -c nn_lint.c
	$(CXX) $(CXXFLAGS) -c nn_lint.cxx
	rm -f nn_lint.c nn_lint.cxx nn_lint.o

nn_test: nn_test.o
	$(CC) $^ -lm -o nn_test

test: nn_test
	./nn_test

clean:
	rm -f nn_test.o nn_test
