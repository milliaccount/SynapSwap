/**
 * @file simple_usage.c
 * @brief Standard usage example of SynapSwap for AI inference
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "synapswap.h"

// Monitoring dashboard (defined in metrics.c)
extern void metrics_display_current();

int main() {
    printf("\033[1;35m=== SynapSwap : AI Inference Demo ===\033[0m\n\n");

    // 1. Initialization: simulate 2 GB VRAM
    // Deliberately over-allocate to trigger swapping
    if (synapswap_init(2ULL * 1024 * 1024 * 1024, true) != SS_SUCCESS) {
        fprintf(stderr, "Failed to initialize SynapSwap engine.\n");
        return 1;
    }

    // 2. Allocate tensors (layers)
    // Total ~3.25 GB, forcing LRU eviction mid-run
    printf("[1/3] Allocating tensors (Estimated total: 3.25 GB)...\n");
    
    void* embed   = synapswap_malloc(512 * 1024 * 1024, 10, SS_POLICY_LRU, "Embedding");
    void* attn_q  = synapswap_malloc(512 * 1024 * 1024,  8, SS_POLICY_LRU, "Attention_Query");
    void* attn_k  = synapswap_malloc(512 * 1024 * 1024,  8, SS_POLICY_LRU, "Attention_Key");
    void* attn_v  = synapswap_malloc(512 * 1024 * 1024,  8, SS_POLICY_LRU, "Attention_Value");
    void* ffn     = synapswap_malloc(1024 * 1024 * 1024, 7, SS_POLICY_LRU, "FFN_Layer");
    void* head    = synapswap_malloc(256 * 1024 * 1024,  9, SS_POLICY_LRU, "Output_Head");

    // 3. Build computation graph
    printf("[2/3] Registering computation graph dependencies...\n");

    // Node 0 (Embed) points to 1,2,3 (parallel attention heads)
    synapswap_register_dependency(0, embed, 1);
    synapswap_register_dependency(0, embed, 2);
    synapswap_register_dependency(0, embed, 3);

    // Attention heads point to FFN (Node 4)
    synapswap_register_dependency(1, attn_q, 4);
    synapswap_register_dependency(2, attn_k, 4);
    synapswap_register_dependency(3, attn_v, 4);

    // FFN points to output (Node 5)
    synapswap_register_dependency(4, ffn, 5);
    synapswap_register_dependency(5, head, -1); // End of graph

    // 4. Simulate Forward Pass
    printf("[3/3] Starting inference simulation...\n\n");

    void* execution_order[] = {embed, attn_q, attn_k, attn_v, ffn, head};
    const char* names[]     = {"Embedding", "Query", "Key", "Value", "FFN", "Head"};

    for (int i = 0; i < 6; i++) {
        // Notify scheduler of current node to trigger prefetch of next nodes
        synapswap_precompute_hint(i);

        printf("\033[1;32m[RUN]\033[0m Processing layer: \033[1m%s\033[0m\n", names[i]);

        // Blocking access if prefetch hasn't finished
        synapswap_wait_for_data(execution_order[i]);

        // Simulate GPU computation
        usleep(150000);

        // Periodic dashboard display
        if (i % 2 == 1) {
            metrics_display_current();
        }
    }

    // 5. Cleanup
    printf("\n\033[1;33mFinal cleanup...\033[0m\n");
    synapswap_free(embed);
    synapswap_free(attn_q);
    synapswap_free(attn_k);
    synapswap_free(attn_v);
    synapswap_free(ffn);
    synapswap_free(head);

    synapswap_shutdown();
    printf("Program finished.\n");

    return 0;
}
