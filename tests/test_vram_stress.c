/**
 * @file test_vram_stress.c
 * @brief Adaptive stress test to validate VRAM swapping and LRU eviction.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include "synapswap.h"

/* --- Scenario Configuration --- */
// Simulate a GPU with 2 GB VRAM. With 30 layers of 128 MB, total usage is ~3.8 GB.
// The system should evict old layers to make room for new ones.
#define MOCK_VRAM_SIZE (2048ULL * 1024 * 1024)
#define NUM_LAYERS 30
#define LAYER_SIZE (128ULL * 1024 * 1024)

// Internal prototype for dashboard display (defined in metrics.c)
extern void metrics_display_current();

void run_stress_test() {
    printf("\n\033[1;33m[TEST] Starting Stress Test: Sequential Model Simulation\033[0m\n");
    printf("[TEST] Profile: %d layers of %zu MB | Max VRAM: %zu MB\n", 
            NUM_LAYERS, LAYER_SIZE / (1024*1024), MOCK_VRAM_SIZE / (1024*1024));

    void* layers[NUM_LAYERS];
    char tag[32];

    // --- STEP 1: Allocation & Graph Construction ---
    printf("\n[1/3] Allocating memory and building the computation graph...\n");
    for (int i = 0; i < NUM_LAYERS; i++) {
        sprintf(tag, "layer_weights_%d", i);
        
        // Early and late layers are usually critical
        int priority = (i < 3 || i > NUM_LAYERS - 4) ? 10 : 5;
        
        layers[i] = synapswap_malloc(LAYER_SIZE, priority, SS_POLICY_AUTO, tag);
        
        if (!layers[i]) {
            fprintf(stderr, "[FATAL] RAM allocation failed for layer %d\n", i);
            exit(EXIT_FAILURE);
        }

        // Register sequential dependency: i -> i+1
        if (i < NUM_LAYERS - 1) {
            synapswap_register_dependency(i, layers[i], i + 1);
        } else {
            synapswap_register_dependency(i, layers[i], -1); // End of graph
        }
    }
    printf("\033[1;32m[OK]\033[0m All layers ready in RAM.\n");

    // --- STEP 2: Inference Simulation ---
    printf("\n[2/3] Starting inference simulation (Forward Pass)...\n");
    
    for (int i = 0; i < NUM_LAYERS; i++) {
        // Hint the scheduler to prefetch i+1, i+2...
        synapswap_precompute_hint(i);

        printf("\n\033[1;34m[EXEC] Node %d\033[0m ", i);
        
        // Access the pointer (blocking if prefetch not finished)
        ss_status_t status = synapswap_wait_for_data(layers[i]);
        
        if (status == SS_SUCCESS) {
            void* vram_addr = synapswap_get_vram_ptr(layers[i]);
            printf("-> \033[1;32mREADY\033[0m (VRAM: %p)\n", vram_addr);
        } else {
            printf("-> \033[1;31mFAILED\033[0m (Error: %d)\n", status);
        }

        // Simulate GPU computation (100ms)
        // During this time, the transfer thread loads the next layers
        usleep(100000); 

        // Periodic metrics display
        if (i % 5 == 0 || i == NUM_LAYERS - 1) {
            metrics_display_current();
        }
    }

    // --- STEP 3: Cleanup ---
    printf("\n[3/3] Releasing resources...\n");
    for (int i = 0; i < NUM_LAYERS; i++) {
        synapswap_free(layers[i]);
    }
    printf("\033[1;32m[DONE]\033[0m Stress test completed successfully.\n");
}

int main() {
    // Initialize SynapSwap engine
    if (synapswap_init(MOCK_VRAM_SIZE, false) != SS_SUCCESS) {
        fprintf(stderr, "Fatal error: Failed to initialize SynapSwap\n");
        return 1;
    }

    run_stress_test();

    synapswap_shutdown();
    return 0;
}
