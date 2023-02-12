CC = g++
CFLAGS = -std=c++20 -Wall -Wextra -MMD
BUILD_DIR = ./build
EXE = main
SRC = $(wildcard *.cpp)
OBJ = ${SRC:.cpp=.o}
DEPS = ${OBJ:.o=.d}
DIR_OBJ = $(addprefix $(BUILD_DIR)/, $(OBJ))
DIR_DEPS = $(addprefix $(BUILD_DIR)/, $(DEPS))

.PHONY: all clean

all: $(EXE)

-include $(DIR_DEPS)

$(EXE): $(DIR_OBJ)
	$(CC) $(DIR_OBJ) $(CFLAGS) -o $(EXE)

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(BUILD_DIR)
	$(CC) $< $(CFLAGS) -g -c -o $@

clean:
	@rm -r $(EXE) $(BUILD_DIR)
