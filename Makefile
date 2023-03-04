CC = g++-11
NVCC = nvcc
CFLAGS = -std=c++20 -Wall -Wextra -MMD -O2
NVCCLAGS = --std=c++11 -MMD -O2 
LDFLAGS = -L/usr/local/cuda-10.2/lib64/ -lcuda -lcudart -lcublas_static -lcublasLt_static -lculibos
BUILD_DIR = ./build
CPP_BUILD_DIR = cpp
CUDA_BUILD_DIR = cuda
EXE = main
CPP_SRC = $(wildcard *.cpp)
CUDA_SRC = $(wildcard *.cu)
CPP_OBJ = ${CPP_SRC:.cpp=.o}
CUDA_OBJ = ${CUDA_SRC:.cu=.o}
CPP_DEPS = ${CPP_OBJ:.o=.d}
CUDA_DEPS = ${CUDA_OBJ:.o=.d}
DIR_CPP_OBJ = $(addprefix $(BUILD_DIR)/$(CPP_BUILD_DIR)/, $(CPP_OBJ))
DIR_CUDA_OBJ = $(addprefix $(BUILD_DIR)/$(CUDA_BUILD_DIR)/, $(CUDA_OBJ))
DIR_CPP_DEPS = $(addprefix $(BUILD_DIR)/, $(CPP_DEPS))
DIR_CUDA_DEPS = $(addprefix $(BUILD_DIR)/, $(CUDA_DEPS))

.PHONY: all clean

all: $(EXE)

-include $(DIR_CPP_DEPS) $(DIR_CUDA_DEPS)

$(EXE): $(DIR_CPP_OBJ) $(DIR_CUDA_OBJ)
	$(CC) $(DIR_CPP_OBJ) $(DIR_CUDA_OBJ) $(CFLAGS) $(LDFLAGS) -o $(EXE)

$(BUILD_DIR)/$(CPP_BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BUILD_DIR)/$(CPP_BUILD_DIR)
	$(CC) $< $(CFLAGS) -c -o $@

$(BUILD_DIR)/$(CUDA_BUILD_DIR)/%.o: %.cu
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BUILD_DIR)/$(CUDA_BUILD_DIR)
	$(NVCC) $< $(NVCCLAGS) -c -o $@

clean:
	@rm -r $(EXE) $(BUILD_DIR)
