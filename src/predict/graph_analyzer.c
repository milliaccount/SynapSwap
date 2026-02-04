/**
 * @file graph_analyzer.c
 * @brief Predictive compute graph analysis and urgency scoring
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "internal_alloc.h"
#include "synapswap.h"

extern ss_context_t g_ctx;
extern ss_status_t internal_load_to_vram(ss_memory_block_t* block);

/* --- Lookahead Structures --- */

typedef struct {
    int node_id;
    float probability;
    int distance;
} ss_lookahead_t;

/**
 * @brief Calculates the urgency of future transfers.
 * Uses a probability propagation algorithm to anticipate memory "hot spots".
 */
void analyze_memory_patterns(int start_node_id) {
    pthread_mutex_lock(&g_ctx.graph_mutex);
    
    // 1. Find the starting point in the graph
    ss_graph_node_t* start_node = g_ctx.graph_head;
    while(start_node && start_node->node_id != start_node_id) {
        start_node = start_node->next;
    }
    
    if (!start_node) {
        pthread_mutex_unlock(&g_ctx.graph_mutex);
        return;
    }

    // 2. Traversal queue (BFS) for predictive analysis
    ss_lookahead_t queue[32]; 
    int head = 0, tail = 0;

    // Start analysis from the current node
    queue[tail++] = (ss_lookahead_t){start_node_id, 1.0f, 0};

    while (head < tail) {
        ss_lookahead_t current = queue[head++];
        
        // Find the corresponding node
        ss_graph_node_t* node = g_ctx.graph_head;
        while(node && node->node_id != current.node_id) node = node->next;

        // Limit depth to avoid saturating the PCIe bus unnecessarily
        if (!node || current.distance >= 5) continue;

        // 3. Scoring and prefetch triggering
        float branch_prob = current.probability / (node->next_count > 0 ? node->next_count : 1);

        for (int i = 0; i < node->next_count; i++) {
            ss_graph_node_t* next_node = g_ctx.graph_head;
            while(next_node && next_node->node_id != node->next_nodes[i]) next_node = next_node->next;

            if (next_node) {
                // For each data block required by this future node
                for (int b_idx = 0; b_idx < next_node->blocks_count; b_idx++) {
                    ss_memory_block_t* b = next_node->required_blocks[b_idx];
                    
                    pthread_mutex_lock(&b->mutex);
                    
                    // Urgency formula: (Traversal probability / Temporal distance) * Intrinsic priority
                    float urgency = (branch_prob / (float)(current.distance + 1)) * (float)b->priority;

                    // Trigger threshold
                    if (urgency > 0.4f && (b->state == MEM_STATE_RAM_ONLY || b->state == MEM_STATE_EVICTED)) {
                        pthread_mutex_unlock(&b->mutex);
                        
                        // Actual asynchronous transfer trigger
                        internal_load_to_vram(b);
                        
                        // Update prefetch metrics
                        pthread_mutex_lock(&g_ctx.stats_mutex);
                        g_ctx.stats.total_access_count++;
                        pthread_mutex_unlock(&g_ctx.stats_mutex);
                    } else {
                        pthread_mutex_unlock(&b->mutex);
                    }
                }

                // 4. Propagate to next nodes (if probability is high enough)
                if (tail < 32 && branch_prob > 0.1f) {
                    queue[tail++] = (ss_lookahead_t){next_node->node_id, branch_prob, current.distance + 1};
                }
            }
        }
    }
    
    pthread_mutex_unlock(&g_ctx.graph_mutex);
}

/**
 * @brief Cache policy optimizer.
 * Adjusts the lookahead depth based on the local topology.
 */
void optimize_prefetch_schedule(int current_node_id) {
    // This function can be extended to adjust g_ctx.vram_limit
    // or change eviction policies on-the-fly.
    
    pthread_mutex_lock(&g_ctx.graph_mutex);
    ss_graph_node_t* node = g_ctx.graph_head;
    while(node && node->node_id != current_node_id) node = node->next;

    if (node && node->next_count > 4) {
        // Highly branched architecture (e.g., MoE)
        // Strategy: Reduce lookahead depth to avoid loading unnecessary branches
        // Increase priority for commonly used weights
    }
    pthread_mutex_unlock(&g_ctx.graph_mutex);
}
