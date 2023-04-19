CC = g++-11
NVCC = nvcc
CFLAGS = -std=c++20 -Wall -Wextra -MMD -g #-O2
NVCCLAGS = --std=c++11 -MMD -O2 
LDFLAGS = -L/usr/local/cuda-10.2/lib64/ -lcuda -lcudart -lcublas_static -lcublasLt_static -lculibos
DEPLOY_DIR = sparse_android_ML/app/src/main/cpp/copied
SRC_DIR = cpp_src
BUILD_DIR = build
CPP_BUILD_DIR = cpp
CUDA_BUILD_DIR = cuda
EXE = main
CPP_SRC = $(wildcard $(SRC_DIR)/*.cpp)
CUDA_SRC = $(wildcard $(SRC_DIR)/*.cu)
CPP_STRIPPED = ${CPP_SRC:$(SRC_DIR)/%=%}
CUDA_STRIPPED = ${CUDA_SRC:$(SRC_DIR)/%=%}
CPP_OBJ = ${CPP_STRIPPED:.cpp=.o}
CUDA_OBJ = ${CUDA_STRIPPED:.cu=.o}
CPP_DEPS = ${CPP_OBJ:.o=.d}
CUDA_DEPS = ${CUDA_OBJ:.o=.d}
DIR_CPP_OBJ = $(addprefix $(BUILD_DIR)/$(CPP_BUILD_DIR)/, $(CPP_OBJ))
DIR_CUDA_OBJ = $(addprefix $(BUILD_DIR)/$(CUDA_BUILD_DIR)/, $(CUDA_OBJ))
DIR_CPP_DEPS = $(addprefix $(BUILD_DIR)/, $(CPP_DEPS))
DIR_CUDA_DEPS = $(addprefix $(BUILD_DIR)/, $(CUDA_DEPS))
DEPLOY_FILES = base.h base.cpp dense.h dense.cpp models.h models.cpp enums.h
COPY_SRC = $(addprefix $(SRC_DIR)/, $(DEPLOY_FILES))

.PHONY: all multi_threaded clean deploy

all:
	$(MAKE) -j8 multi_threaded

multi_threaded: $(EXE)

-include $(DIR_CPP_DEPS) $(DIR_CUDA_DEPS)

$(EXE): $(DIR_CPP_OBJ) $(DIR_CUDA_OBJ)
	$(CC) $(DIR_CPP_OBJ) $(DIR_CUDA_OBJ) $(CFLAGS) $(LDFLAGS) -o $(EXE)

$(BUILD_DIR)/$(CPP_BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BUILD_DIR)/$(CPP_BUILD_DIR)
	$(CC) $< $(CFLAGS) -c -o $@

$(BUILD_DIR)/$(CUDA_BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BUILD_DIR)/$(CUDA_BUILD_DIR)
	$(NVCC) $< $(NVCCLAGS) -c -o $@

deploy:
	$(foreach file, $(COPY_SRC), cp $(file) $(DEPLOY_DIR);)

$(DEPLOY_DIR)/%: $(COPY_SRC)
	@echo "Copying $< to $(DEPLOY_DIR)/"
	@cp -rf $< $(DEPLOY_DIR)/

clean:
	@rm -r $(EXE) $(BUILD_DIR)
