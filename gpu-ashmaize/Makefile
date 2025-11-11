# GPU-AshMaize Makefile
# Build system for CUDA unit tests

NVCC := nvcc
CUDA_ARCH := sm_75
CUDA_FLAGS := -arch=$(CUDA_ARCH) -O3 --std=c++17 -Xcompiler -Wall
CUDA_DIR := cuda
TESTS_DIR := tests
BUILD_DIR := target/cuda_tests

# Test executables (production test suite)
TEST_BLAKE2B := $(BUILD_DIR)/test_blake2b
TEST_BLAKE2B_VERIFY := $(BUILD_DIR)/test_blake2b_verify
TEST_ARGON2 := $(BUILD_DIR)/test_argon2

# Benchmark executable
BENCHMARK := $(BUILD_DIR)/benchmark

.PHONY: all clean test test-blake2b test-blake2b-verify test-argon2 bench

all: $(TEST_BLAKE2B) $(TEST_BLAKE2B_VERIFY) $(TEST_ARGON2) $(BENCHMARK)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Blake2b test  
$(TEST_BLAKE2B): $(TESTS_DIR)/blake2b_test.cu $(CUDA_DIR)/blake2b.cu $(CUDA_DIR)/blake2b.cuh $(CUDA_DIR)/common.cuh | $(BUILD_DIR)
	@echo "Building Blake2b tests..."
	$(NVCC) $(CUDA_FLAGS) \
		-rdc=true \
		-I. \
		$(CUDA_DIR)/blake2b.cu \
		$(TESTS_DIR)/blake2b_test.cu \
		-o $(TEST_BLAKE2B)

# Blake2b verification test
$(TEST_BLAKE2B_VERIFY): $(TESTS_DIR)/blake2b_verify.cu $(CUDA_DIR)/blake2b.cu $(CUDA_DIR)/blake2b.cuh $(CUDA_DIR)/common.cuh | $(BUILD_DIR)
	@echo "Building Blake2b verification tests..."
	$(NVCC) $(CUDA_FLAGS) \
		-rdc=true \
		-I. \
		$(CUDA_DIR)/blake2b.cu \
		$(TESTS_DIR)/blake2b_verify.cu \
		-o $(TEST_BLAKE2B_VERIFY)

# Argon2H' test
$(TEST_ARGON2): $(TESTS_DIR)/argon2_test.cu $(CUDA_DIR)/argon2.cu $(CUDA_DIR)/blake2b.cu $(CUDA_DIR)/argon2.cuh $(CUDA_DIR)/blake2b.cuh $(CUDA_DIR)/common.cuh | $(BUILD_DIR)
	@echo "Building Argon2H' tests..."
	$(NVCC) $(CUDA_FLAGS) \
		-rdc=true \
		-I. \
		$(CUDA_DIR)/blake2b.cu \
		$(CUDA_DIR)/argon2.cu \
		$(TESTS_DIR)/argon2_test.cu \
		-o $(TEST_ARGON2)

# Benchmark executable
$(BENCHMARK): $(CUDA_DIR)/benchmark.cu $(CUDA_DIR)/kernel.cu $(CUDA_DIR)/instructions.cu $(CUDA_DIR)/vm.cu $(CUDA_DIR)/blake2b.cu $(CUDA_DIR)/argon2.cu $(CUDA_DIR)/kernel.cuh $(CUDA_DIR)/instructions.cuh $(CUDA_DIR)/vm.cuh $(CUDA_DIR)/blake2b.cuh $(CUDA_DIR)/common.cuh | $(BUILD_DIR)
	@echo "Building Benchmark..."
	$(NVCC) $(CUDA_FLAGS) \
		-rdc=true \
		-I. \
		$(CUDA_DIR)/blake2b.cu \
		$(CUDA_DIR)/argon2.cu \
		$(CUDA_DIR)/instructions.cu \
		$(CUDA_DIR)/vm.cu \
		$(CUDA_DIR)/benchmark.cu \
		-o $(BENCHMARK)

# Run specific test
test-blake2b: $(TEST_BLAKE2B)
	@echo "=== Running Blake2b Tests ==="
	@$(TEST_BLAKE2B)

# Run Blake2b verification against Python reference
test-blake2b-verify: $(TEST_BLAKE2B_VERIFY)
	@echo "=== Running Blake2b Verification Tests ==="
	@$(TEST_BLAKE2B_VERIFY)

# Run Argon2H' tests
test-argon2: $(TEST_ARGON2)
	@echo "=== Running Argon2H' Tests ==="
	@$(TEST_ARGON2)

# Run all tests
test: test-blake2b test-blake2b-verify test-argon2
	@echo ""
	@echo "=== All CUDA Tests Complete ==="
	@echo ""
	@echo "Note: Additional integration tests available via Rust:"
	@echo "  cargo test --release  # Run Rust integration tests"
	@echo "  cargo run --release --example cpu_vs_gpu_benchmark  # Performance tests"

# Run benchmark
bench: $(BENCHMARK)
	@echo "=== Running GPU Benchmark ==="
	@$(BENCHMARK)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	@echo "Cleaned build artifacts"

# Help
help:
	@echo "GPU-AshMaize Test Makefile"
	@echo ""
	@echo "Production Test Suite (CUDA):"
	@echo "  all                - Build all tests (default)"
	@echo "  test               - Run all CUDA tests"
	@echo "  test-blake2b       - Run Blake2b unit tests (28 tests)"
	@echo "  test-blake2b-verify - Run Blake2b verification vs Python"
	@echo "  test-argon2        - Run Argon2H' tests (13 tests)"
	@echo "  bench              - Run GPU performance benchmark"
	@echo "  clean              - Remove build artifacts"
	@echo "  help               - Show this help"
	@echo ""
	@echo "Integration Tests (Rust):"
	@echo "  Run via Cargo from project root:"
	@echo "    cargo test --release              # CPU vs GPU equivalence"
	@echo "    cargo run --example cpu_vs_gpu_benchmark  # Performance comparison"
	@echo ""
	@echo "Variables:"
	@echo "  CUDA_ARCH    - CUDA architecture (default: sm_75)"
	@echo "  NVCC         - CUDA compiler (default: nvcc)"
	@echo ""
	@echo "Example:"
	@echo "  make CUDA_ARCH=sm_86 test"
	@echo ""
	@echo "Archived Tests:"
	@echo "  7 test files moved to /ash-docs-gpu/ (see MOVED_TESTS_README.md)"
	@echo "  - Debugging tests: test_exact_seed, vm_determinism_debug"
	@echo "  - Performance demos: test_large_roms, test_early_exit"
	@echo "  - Detailed unit tests: instructions_test, vm_test, kernel_test"
