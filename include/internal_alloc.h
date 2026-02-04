/**
 * @file internal_alloc.h
 * @brief Internal structures and SynapSwap memory management engine
 */

#ifndef INTERNAL_ALLOC_H
#define INTERNAL_ALLOC_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <pthread.h>
#include "synapswap.h"

/* --- Internal Configuration --- */
#define SS_HASH_SIZE 1024        // Hash table size
#define SS_MAX_TAG_LEN 64
#define SS_ALIGNMENT 256         // Optimal alignment for DMA (Nvidia/AMD)

/* --- Backend Abstraction (CUDA/ROCm/CPU) --- */
#ifdef USE_CUDA
    #include <cuda_runtime.h>
    typedef cudaStream_t ss_stream_t;
    typedef cudaEvent_t  ss_event_t;
#else
    typedef void* ss_stream_t;
    typedef void* ss_event_t;
#endif

/* --- Memory States --- */
typedef enum {
    MEM_STATE_RAM_ONLY,      // Only in system RAM
    MEM_STATE_VRAM_ONLY,     // Only in VRAM (Static weights)
    MEM_STATE_SWAPPING_IN,   // Host -> Device transfer in progress
    MEM_STATE_SWAPPING_OUT,  // Device -> Host transfer in progress
    MEM_STATE_RESIDENT,      // Present in VRAM (synchronized)
    MEM_STATE_EVICTED        // Evicted (RAM valid, VRAM freed)
} ss_mem_state_t;

/* --- Internal Memory Block --- */
typedef struct ss_memory_block {
    void* vram_ptr;          // Actual GPU address
    void* ram_ptr;           // Host address (pinned memory if possible)
    size_t size;
    int priority;
    ss_mem_state_t state;
    ss_policy_t policy;

    uint32_t access_count;
    uint64_t last_access_tick; // LRU timestamp (Global Tick)
    char tag[SS_MAX_TAG_LEN];

    ss_event_t ready_event;   // Hardware synchronization event
    pthread_mutex_t mutex;    // Lock for the block's state machine
    struct ss_memory_block* next_hash; 
} ss_memory_block_t;

/* --- Compute Graph & Prediction --- */
typedef struct ss_graph_node {
    int node_id;
    ss_memory_block_t** required_blocks; // List of required tensors
    int blocks_count;
    int* next_nodes;           // Transition probabilities
    int next_count;
    struct ss_graph_node* next;
} ss_graph_node_t;

/* --- Transfer Requests --- */
typedef struct ss_transfer_request {
    ss_memory_block_t* block;
    bool to_vram;
    uint64_t request_tick;     // For prioritizing urgent transfers
    struct ss_transfer_request* next;
} ss_transfer_request_t;

/* --- Global Context (Singleton) --- */
typedef struct {
    // Memory management
    size_t vram_limit;
    size_t vram_allocated;
    size_t ram_swap_allocated;

    // Backend info
    ss_stream_t dma_stream;    // Dedicated async swap stream (overlap)
    bool use_unified_memory;

    // Hash table (O(1) lookup)
    ss_memory_block_t* block_hash_table[SS_HASH_SIZE];
    pthread_rwlock_t table_lock;

    // Prediction graph
    ss_graph_node_t* graph_head;
    pthread_mutex_t graph_mutex;
    uint64_t global_tick;      // Logical clock incremented on each access

    // Transfer queue
    ss_transfer_request_t *queue_head, *queue_tail;
    pthread_mutex_t queue_mutex;
    pthread_cond_t  queue_cond;
    pthread_t transfer_worker;
    bool shutdown_flag;

    // Metrics (real-time)
    ss_stats_t stats;
    pthread_mutex_t stats_mutex;
} ss_context_t;

/* --- Internal Core Prototypes --- */

/**
 * @brief Finds a memory block from a RAM address.
 */
ss_memory_block_t* _ss_find_block(void* ptr);

/**
 * @brief Registers a new memory block in the system.
 */
void _ss_register_block(ss_memory_block_t* block);

/**
 * @brief LRU + Priority eviction algorithm.
 * Evicts blocks until needed_size is available.
 */
ss_status_t _ss_evict_vram(size_t needed_size);

/**
 * @brief Physical copy interface (Implemented in backend-specific code).
 */
ss_status_t _ss_phys_alloc_vram(ss_memory_block_t* block);
ss_status_t _ss_phys_free_vram(ss_memory_block_t* block);
ss_status_t _ss_phys_copy_async(ss_memory_block_t* block, bool to_vram);

#endif /* INTERNAL_ALLOC_H */
