/**
 * @file scheduler.c
 * @brief Predictive scheduling algorithm and compute graph management
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "internal_alloc.h"
#include "synapswap.h"

extern ss_context_t g_ctx;
extern ss_status_t internal_load_to_vram(ss_memory_block_t* block);
extern ss_status_t _ss_evict_vram(size_t needed_size);

/* --- Graph Utilities --- */

/**
 * @brief Finds a node in the graph (O(N) but the graph is small < 2000 nodes)
 */
static ss_graph_node_t* _ss_find_graph_node(int node_id) {
    pthread_mutex_lock(&g_ctx.graph_mutex);
    ss_graph_node_t* curr = g_ctx.graph_head;
    while (curr) {
        if (curr->node_id == node_id) {
            pthread_mutex_unlock(&g_ctx.graph_mutex);
            return curr;
        }
        curr = curr->next;
    }
    pthread_mutex_unlock(&g_ctx.graph_mutex);
    return NULL;
}

/* --- Predictive Prefetch Engine --- */

/**
 * @brief Analyzes future needs and pre-allocates VRAM.
 * @param current_node_id ID of the current execution node.
 */
void _ss_prefetch_engine(int current_node_id) {
    ss_graph_node_t* start_node = _ss_find_graph_node(current_node_id);
    if (!start_node) return;

    // Prediction horizon: how many layers to look ahead?
    // In production, adjust according to available PCIe bandwidth.
    int lookahead_horizon = 4; 
    
    ss_graph_node_t* curr = start_node;
    for (int d = 0; d < lookahead_horizon; d++) {
        // For each probable future node
        for (int i = 0; i < curr->next_count; i++) {
            ss_graph_node_t* neighbor = _ss_find_graph_node(curr->next_nodes[i]);
            if (!neighbor) continue;

            // For each block (tensor) required by this node
            for (int b_idx = 0; b_idx < neighbor->blocks_count; b_idx++) {
                ss_memory_block_t* block = neighbor->required_blocks[b_idx];
                
                pthread_mutex_lock(&block->mutex);
                if (block->state == MEM_STATE_RAM_ONLY || block->state == MEM_STATE_EVICTED) {
                    
                    // Attempt eviction if VRAM is full
                    if (g_ctx.vram_allocated + block->size > g_ctx.vram_limit) {
                        pthread_mutex_unlock(&block->mutex);
                        if (_ss_evict_vram(block->size) != SS_SUCCESS) {
                            continue; // Eviction failed, skip this prefetch
                        }
                        pthread_mutex_lock(&block->mutex);
                    }

                    // Launch asynchronous loading
                    // Note: NEVER block here
                    internal_load_to_vram(block);
                    
                    // Adjust transfer priority: closer = more urgent
                    block->priority = (10 - d); 
                }
                pthread_mutex_unlock(&block->mutex);
            }
            // Follow the first path for sequential prediction
            if (i == 0) curr = neighbor; 
        }
        if (!curr || curr->next_count == 0) break;
    }
}

/* --- Graph API Implementation --- */

SS_API void synapswap_register_dependency(int node_id, void* data_ptr, int next_node_id) {
    ss_memory_block_t* block = _ss_find_block(data_ptr);
    if (!block) return;

    ss_graph_node_t* node = _ss_find_graph_node(node_id);

    pthread_mutex_lock(&g_ctx.graph_mutex);
    if (!node) {
        node = (ss_graph_node_t*)calloc(1, sizeof(ss_graph_node_t));
        node->node_id = node_id;
        node->next = g_ctx.graph_head;
        g_ctx.graph_head = node;
    }

    // Add block to the list of required blocks for this node
    node->blocks_count++;
    node->required_blocks = (ss_memory_block_t**)realloc(
        node->required_blocks, 
        node->blocks_count * sizeof(ss_memory_block_t*)
    );
    node->required_blocks[node->blocks_count - 1] = block;

    // Add transition to the next node
    bool exists = false;
    for(int i = 0; i < node->next_count; i++) {
        if (node->next_nodes[i] == next_node_id) exists = true;
    }

    if (!exists && next_node_id != -1) {
        node->next_count++;
        node->next_nodes = (int*)realloc(node->next_nodes, node->next_count * sizeof(int));
        node->next_nodes[node->next_count - 1] = next_node_id;
    }
    pthread_mutex_unlock(&g_ctx.graph_mutex);
}

SS_API void synapswap_precompute_hint(int node_id) {
    pthread_mutex_lock(&g_ctx.stats_mutex);
    g_ctx.global_tick++; // Advance logical clock
    pthread_mutex_unlock(&g_ctx.stats_mutex);

    // Trigger the scheduling engine
    _ss_prefetch_engine(node_id);
}
