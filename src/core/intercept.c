/**
 * @file intercept.c
 * @brief Interception layer and RAM/VRAM decision logic
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <sched.h>
#include "internal_alloc.h"
#include "synapswap.h"

/* --- Aligned memory compatibility macros --- */
#ifdef _WIN32
    #include <malloc.h>
    #define SS_ALIGNED_ALLOC(size, align) _aligned_malloc(size, align)
    #define SS_ALIGNED_FREE(ptr) _aligned_free(ptr)
#else
    #include <stdlib.h>
    #define SS_ALIGNED_ALLOC(size, align) aligned_alloc(align, size)
    #define SS_ALIGNED_FREE(ptr) free(ptr)
#endif

// External declarations (implemented in scheduler.c and transfer_engine.c)
extern void* internal_transfer_thread(void* arg);
extern ss_status_t internal_load_to_vram(ss_memory_block_t* block);
extern void metrics_record_access(bool hit, bool was_prefetch);

ss_context_t g_ctx;
static bool g_initialized = false;

/* --- Hash Table Management --- */

static uint32_t _hash_ptr(void* ptr) {
    return ((uintptr_t)ptr >> 4) % SS_HASH_SIZE;
}

ss_memory_block_t* _ss_find_block(void* ptr) {
    if (!ptr) return NULL;
    uint32_t index = _hash_ptr(ptr);
    
    pthread_rwlock_rdlock(&g_ctx.table_lock);
    ss_memory_block_t* curr = g_ctx.block_hash_table[index];
    while (curr) {
        if (curr->ram_ptr == ptr || curr->vram_ptr == ptr) {
            pthread_rwlock_unlock(&g_ctx.table_lock);
            return curr;
        }
        curr = curr->next_hash;
    }
    pthread_rwlock_unlock(&g_ctx.table_lock);
    return NULL;
}

void _ss_register_block(ss_memory_block_t* block) {
    uint32_t index = _hash_ptr(block->ram_ptr);
    pthread_rwlock_wrlock(&g_ctx.table_lock);
    block->next_hash = g_ctx.block_hash_table[index];
    g_ctx.block_hash_table[index] = block;
    pthread_rwlock_unlock(&g_ctx.table_lock);
}

/* --- Public API Implementation --- */

ss_status_t synapswap_init(size_t vram_limit, bool use_unified_memory) {
    if (g_initialized) return SS_SUCCESS;

    memset(&g_ctx, 0, sizeof(ss_context_t));
    
    pthread_rwlock_init(&g_ctx.table_lock, NULL);
    pthread_mutex_init(&g_ctx.queue_mutex, NULL);
    pthread_cond_init(&g_ctx.queue_cond, NULL);
    pthread_mutex_init(&g_ctx.stats_mutex, NULL);
    pthread_mutex_init(&g_ctx.graph_mutex, NULL);

    // If vram_limit is 0, could query CUDA device properties here
    g_ctx.vram_limit = (vram_limit > 0) ? vram_limit : (4ULL * 1024 * 1024 * 1024);
    g_ctx.use_unified_memory = use_unified_memory;
    g_ctx.shutdown_flag = false;
    g_ctx.global_tick = 1;

    if (pthread_create(&g_ctx.transfer_worker, NULL, internal_transfer_thread, &g_ctx) != 0) {
        return SS_ERR_INITIALIZATION;
    }

    g_initialized = true;
    printf("[SynapSwap] Engine initialized (Limit: %zu MB)\n", g_ctx.vram_limit / (1024*1024));
    return SS_SUCCESS;
}

void* synapswap_malloc(size_t size, int priority, ss_policy_t policy, const char* tag) {
    if (!g_initialized) return NULL;

    // Align memory for optimal DMA
    size_t aligned_size = (size + SS_ALIGNMENT - 1) & ~(SS_ALIGNMENT - 1);
    ss_memory_block_t* block = (ss_memory_block_t*)calloc(1, sizeof(ss_memory_block_t));
    if (!block) return NULL;

    // Allocate RAM (Host Pinned Memory ideal for CUDA)
    block->ram_ptr = SS_ALIGNED_ALLOC(aligned_size, SS_ALIGNMENT);
    if (!block->ram_ptr) {
        free(block);
        return NULL;
    }

    block->size = aligned_size;
    block->priority = (priority < 0) ? 0 : (priority > 10 ? 10 : priority);
    block->policy = policy;
    block->state = MEM_STATE_RAM_ONLY;
    block->last_access_tick = 0;
    
    if (tag) strncpy(block->tag, tag, SS_MAX_TAG_LEN - 1);
    pthread_mutex_init(&block->mutex, NULL);

    _ss_register_block(block);

    // Immediate preloading if necessary
    if (policy == SS_POLICY_STRICT_VRAM || priority >= 9) {
        internal_load_to_vram(block);
    }

    return block->ram_ptr;
}

ss_status_t synapswap_wait_for_data(void* ptr) {
    ss_memory_block_t* block = _ss_find_block(ptr);
    if (!block) return SS_ERR_INVALID_PTR;

    pthread_mutex_lock(&block->mutex);

    // Update LRU tick
    pthread_mutex_lock(&g_ctx.stats_mutex);
    block->last_access_tick = ++g_ctx.global_tick;
    pthread_mutex_unlock(&g_ctx.stats_mutex);

    // 1. Demand Paging: if data not present and not requested
    if (block->state == MEM_STATE_RAM_ONLY || block->state == MEM_STATE_EVICTED) {
        pthread_mutex_unlock(&block->mutex);
        
        // Cache miss
        metrics_record_access(false, false);
        
        // Force loading to VRAM
        internal_load_to_vram(block);
        
        pthread_mutex_lock(&block->mutex);
    }

    // 2. Wait for transfer completion
    while (block->state == MEM_STATE_SWAPPING_IN) {
        pthread_mutex_unlock(&block->mutex);
        sched_yield(); 
        pthread_mutex_lock(&block->mutex);
    }
    
    if (block->state == MEM_STATE_RESIDENT) {
        metrics_record_access(true, false);
        pthread_mutex_unlock(&block->mutex);
        return SS_SUCCESS;
    }

    pthread_mutex_unlock(&block->mutex);
    return SS_ERR_NOT_FOUND;
}

void* synapswap_get_vram_ptr(void* ptr) {
    ss_memory_block_t* block = _ss_find_block(ptr);
    if (!block || block->state != MEM_STATE_RESIDENT) return NULL;
    return block->vram_ptr;
}

void synapswap_free(void* ptr) {
    ss_memory_block_t* block = _ss_find_block(ptr);
    if (!block) return;

    // Safely remove from hash table
    uint32_t index = _hash_ptr(ptr);
    pthread_rwlock_wrlock(&g_ctx.table_lock);
    ss_memory_block_t** curr = &g_ctx.block_hash_table[index];
    while (*curr) {
        if (*curr == block) {
            *curr = block->next_hash;
            break;
        }
        curr = &((*curr)->next_hash);
    }
    pthread_rwlock_unlock(&g_ctx.table_lock);

    // Hardware free
    pthread_mutex_lock(&block->mutex);
    if (block->vram_ptr) {
        pthread_mutex_lock(&g_ctx.stats_mutex);
        g_ctx.vram_allocated -= block->size;
        pthread_mutex_unlock(&g_ctx.stats_mutex);
        
        // Production: cudaFree(block->vram_ptr);
        free(block->vram_ptr); 
    }
    if (block->ram_ptr) SS_ALIGNED_FREE(block->ram_ptr);
    pthread_mutex_unlock(&block->mutex);
    
    pthread_mutex_destroy(&block->mutex);
    free(block);
}

void synapswap_shutdown(void) {
    g_ctx.shutdown_flag = true;
    pthread_cond_signal(&g_ctx.queue_cond);
    pthread_join(g_ctx.transfer_worker, NULL);
    
    // Final cleanup of locks
    pthread_rwlock_destroy(&g_ctx.table_lock);
    pthread_mutex_destroy(&g_ctx.queue_mutex);
    pthread_cond_destroy(&g_ctx.queue_cond);
    pthread_mutex_destroy(&g_ctx.stats_mutex);
    pthread_mutex_destroy(&g_ctx.graph_mutex);
    
    printf("[SynapSwap] Shutdown complete.\n");
}
