# --- Configuration ---
CC       := gcc
# Note: CXX should be a compiler, not just a context variable
CXX      := g++
# Add -Wextra for more warnings to catch potential issues
CFLAGS   := -I./include -Wall -Wextra -O3 -pthread -DSS_STATIC_BUILD
LDFLAGS  := -pthread -lm

# OS detection for extensions (Windows vs Linux/macOS)
ifeq ($(OS),Windows_NT)
    EXT := .exe
    LIB_EXT := .dll
else
    EXT :=
    LIB_EXT := .so
endif

# --- Source files & objects ---
SRC_DIR  := src
CORE_SRC := $(wildcard $(SRC_DIR)/core/*.c)
PRED_SRC := $(wildcard $(SRC_DIR)/predict/*.c)
TEST_SRC := tests/test_vram_stress.c

SRCS     := $(CORE_SRC) $(PRED_SRC)
OBJS     := $(SRCS:.c=.o)
TEST_OBJ := $(TEST_SRC:.c=.o)

# --- Targets ---
LIB_NAME := libsynapswap$(LIB_EXT)
TARGET   := synapswap_test$(EXT)

.PHONY: all clean help

all: $(TARGET) $(LIB_NAME)

# Build the test executable
$(TARGET): $(OBJS) $(TEST_OBJ)
	@echo "🔗 Linking executable: $@"
	@$(CC) -o $@ $^ $(LDFLAGS)
	@echo "✅ Build complete: ./\033[1;32m$@\033[0m"

# Build the shared library
$(LIB_NAME): $(OBJS)
	@echo "🔨 Building shared library: $@"
	@$(CC) -shared -o $@ $^ $(LDFLAGS)

# Generic rule for object files
%.o: %.c
	@echo "📦 Compiling: $<"
	@$(CC) $(CFLAGS) -c $< -o $@

# --- Maintenance ---
clean:
	@echo "🧹 Cleaning up..."
	@rm -f $(OBJS) $(TEST_OBJ) $(TARGET) $(LIB_NAME)
	@echo "✨ Clean done."

help:
	@echo "SynapSwap Build System"
	@echo "Targets:"
	@echo "  all     : Build the test executable and shared library"
	@echo "  clean   : Remove object files and binaries"
	@echo "  help    : Show this help message"
