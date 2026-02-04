/**
 * @file transfer_engine.c
 * @brief Data movement engine and LRU eviction manager
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include "internal_alloc.h"
#include "synapswap.h"

#ifdef USE_CUDA
    #include <cuda_runtime.h>
#endif

extern ss_context_t g_ctx;

/* --- LRU-Priority Eviction Engine --- */

/**
 * @brief Selects and frees the least useful VRAM blocks.
 * Logic: Lowest priority first, then oldest tick (LRU).
 */
ss_status_t _ss_evict_vram(size_t needed_size) {
    if (needed_size > g_ctx.vram_limit) return SS_ERR_OUT_OF_VRAM;

    while (g_ctx.vram_allocated + needed_size > g_ctx.vram_limit) {
        ss_memory_block_t* victim = NULL;
        int lowest_priority = 11;
        uint64_t oldest_tick = UINT64_MAX;

        pthread_rwlock_rdlock(&g_ctx.table_lock);
        // Iterate over hash table to find the best victim
        for (int i = 0; i < SS_HASH_SIZE; i++) {
            ss_memory_block_t* curr = g_ctx.block_hash_table[i];
            while (curr) {
                // Only consider resident blocks and non-STRICT policy
                if (curr->state == MEM_STATE_RESIDENT && curr->policy != SS_POLICY_STRICT_VRAM) {
                    if (curr->priority < lowest_priority) {
                        lowest_priority = curr->priority;
                        oldest_tick = curr->last_access_tick;
                        victim = curr;
                    } 
                    else if (curr->priority == lowest_priority && curr->last_access_tick < oldest_tick) {
                        oldest_tick = curr->last_access_tick;
                        victim = curr;
                    }
                }
                curr = curr->next_hash;
            }
        }
        pthread_rwlock_unlock(&g_ctx.table_lock);

        if (!victim) return SS_ERR_OUT_OF_VRAM;

        // Physical eviction
        pthread_mutex_lock(&victim->mutex);
        if (victim->vram_ptr) {
            #ifdef USE_CUDA
                cudaFree(victim->vram_ptr);
            #else
                free(victim->vram_ptr);
            #endif
            
            victim->vram_ptr = NULL;
            victim->state = MEM_STATE_EVICTED;
            
            pthread_mutex_lock(&g_ctx.stats_mutex);
            g_ctx.vram_allocated -= victim->size;
            pthread_mutex_unlock(&g_ctx.stats_mutex);
            
            printf("[SynapSwap] EVICT <- %s (LRU Tick: %lu)\n", victim->tag, (unsigned long)victim->last_access_tick);
        }
        pthread_mutex_unlock(&victim->mutex);
    }
    return SS_SUCCESS;
}

/* --- Transfer Queue Management --- */

static void _ss_enqueue_request(ss_memory_block_t* block, bool to_vram) {
    ss_transfer_request_t* req = (ss_transfer_request_t*)malloc(sizeof(ss_transfer_request_t));
    req->block = block;
    req->to_vram = to_vram;
    req->next = NULL;

    pthread_mutex_lock(&g_ctx.queue_mutex);
    if (g_ctx.queue_tail) {
        g_ctx.queue_tail->next = req;
    } else {
        g_ctx.queue_head = req;
    }
    g_ctx.queue_tail = req;
    
    pthread_cond_signal(&g_ctx.queue_cond);
    pthread_mutex_unlock(&g_ctx.queue_mutex);
}

/* --- Physical Transfer API --- */

ss_status_t internal_load_to_vram(ss_memory_block_t* block) {
    pthread_mutex_lock(&block->mutex);
    if (block->state == MEM_STATE_RESIDENT || block->state == MEM_STATE_SWAPPING_IN) {
        pthread_mutex_unlock(&block->mutex);
        return SS_SUCCESS;
    }
    block->state = MEM_STATE_SWAPPING_IN;
    pthread_mutex_unlock(&block->mutex);

    _ss_enqueue_request(block, true);
    return SS_SUCCESS;
}

/* --- Worker Thread (DMA Core) --- */

void* internal_transfer_thread(void* arg) {
    ss_context_t* ctx = (ss_context_t*)arg;

    while (!ctx->shutdown_flag) {
        pthread_mutex_lock(&ctx->queue_mutex);
        while (ctx->queue_head == NULL && !ctx->shutdown_flag) {
            pthread_cond_wait(&ctx->queue_cond, &ctx->queue_mutex);
        }

        if (ctx->shutdown_flag && ctx->queue_head == NULL) {
            pthread_mutex_unlock(&ctx->queue_mutex);
            break;
        }

        ss_transfer_request_t* req = ctx->queue_head;
        ctx->queue_head = req->next;
        if (!ctx->queue_head) ctx->queue_tail = NULL;
        pthread_mutex_unlock(&ctx->queue_mutex);

        ss_memory_block_t* b = req->block;

        if (req->to_vram) {
            // 1. Ensure GPU space
            if (ctx->vram_allocated + b->size > ctx->vram_limit) {
                _ss_evict_vram(b->size);
            }

            // 2. Actual GPU allocation
            #ifdef USE_CUDA
                cudaMalloc(&b->vram_ptr, b->size);
                // Asynchronous transfer (PCIe DMA)
                cudaMemcpyAsync(b->vram_ptr, b->ram_ptr, b->size, cudaMemcpyHostToDevice, ctx->dma_stream);
                cudaStreamSynchronize(ctx->dma_stream); 
            #else
                b->vram_ptr = malloc(b->size);
                memcpy(b->vram_ptr, b->ram_ptr, b->size);
            #endif

            // 3. Update state
            pthread_mutex_lock(&b->mutex);
            b->state = MEM_STATE_RESIDENT;
            pthread_mutex_lock(&ctx->stats_mutex);
            ctx->vram_allocated += b->size;
            ctx->stats.total_transfers++;
            pthread_mutex_unlock(&ctx->stats_mutex);
            pthread_mutex_unlock(&b->mutex);
        }
        
        free(req);
    }
    return NULL;
}
