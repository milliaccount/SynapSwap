/**
 * @file synapswap.h
 * @brief Public API for SynapSwap - Cross-Platform Predictive VRAM Manager.
 */

#ifndef SYNAPSWAP_H
#define SYNAPSWAP_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

/* --- Visibility & DLL Compatibility Macros --- */

#if defined(_WIN32) || defined(__CYGWIN__)
    #if defined(SS_STATIC_BUILD)
        // For MINGW or static build, leave empty to avoid attribute conflicts
        #define SS_API 
    #elif defined(SYNAPSWAP_EXPORTS)
        #define SS_API __declspec(dllexport)
    #else
        #define SS_API __declspec(dllimport)
    #endif
#else
    #if __GNUC__ >= 4
        #define SS_API __attribute__ ((visibility ("default")))
    #else
        #define SS_API
    #endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* --- Types and Enumerations --- */

/**
 * @brief Error codes for the SynapSwap API
 */
typedef enum {
    SS_SUCCESS = 0,
    SS_ERR_INITIALIZATION = -1,  /**< GPU driver or system resources failed */
    SS_ERR_OUT_OF_VRAM     = -2,  /**< GPU memory full (eviction impossible) */
    SS_ERR_OUT_OF_RAM      = -3,  /**< System RAM full */
    SS_ERR_INVALID_PTR     = -4,  /**< Pointer not managed by SynapSwap */
    SS_ERR_NOT_FOUND       = -5,  /**< Data not found in cache */
    SS_ERR_CUDA            = -6,  /**< NVIDIA driver-specific error */
    SS_ERR_UNKNOWN         = -99
} ss_status_t;

/**
 * @brief Memory placement strategies (Cache Policies)
 */
typedef enum {
    SS_POLICY_AUTO = 0,      /**< Dynamic decision (LRU + Prediction) */
    SS_POLICY_STRICT_VRAM,   /**< Locked in VRAM (e.g., critical weights, KV Cache) */
    SS_POLICY_LRU,           /**< Classic Least Recently Used eviction */
    SS_POLICY_VOLATILE,      /**< Freed immediately after synapswap_wait_for_data */
} ss_policy_t;

/**
 * @brief Extended statistics structure for monitoring
 */
typedef struct {
    size_t vram_total_bytes;
    size_t vram_used_bytes;
    size_t ram_swap_used_bytes;
    uint64_t total_transfers;
    uint64_t total_access_count;
    double avg_transfer_latency_ms;
    float prefetch_hit_rate; 
    uint32_t active_blocks;  
} ss_stats_t;

/* --- Lifecycle Functions --- */

/**
 * @brief Initializes the SynapSwap engine.
 * @param vram_limit VRAM limit in bytes.
 * @param verbose Enables transfer logging to console.
 */
SS_API ss_status_t synapswap_init(size_t vram_limit, bool verbose);

/**
 * @brief Releases all resources and stops transfer threads.
 */
SS_API void synapswap_shutdown(void);

/* --- Allocation Functions --- */

/**
 * @brief Allocates a memory block managed by the swap system.
 * @param size Size in bytes.
 * @param priority Priority (0: low, 10: critical).
 * @param policy Eviction policy.
 * @param tag Block name for monitoring purposes.
 */
SS_API void* synapswap_malloc(size_t size, int priority, ss_policy_t policy, const char* tag);

/**
 * @brief Frees a managed memory block.
 */
SS_API void synapswap_free(void* ptr);

/* --- Prediction Engine & Graph --- */

/**
 * @brief Registers a dependency between two compute nodes.
 * Allows the engine to build a dependency graph for prefetching.
 */
SS_API void synapswap_register_dependency(int node_id, void* data_ptr, int next_node_id);

/**
 * @brief Notifies the scheduler of the current compute node ID.
 * Triggers asynchronous prefetching of likely future nodes.
 */
SS_API void synapswap_precompute_hint(int node_id);

/**
 * @brief Synchronization: waits until data is resident in VRAM.
 * @return SS_SUCCESS if the data is ready to be used by the GPU.
 */
SS_API ss_status_t synapswap_wait_for_data(void* ptr);

/**
 * @brief Retrieves the physical GPU address (Device Pointer).
 */
SS_API void* synapswap_get_vram_ptr(void* ptr);

/* --- Monitoring --- */

/**
 * @brief Fills the stats structure with current engine data.
 */
SS_API void synapswap_get_stats(ss_stats_t* stats);

#ifdef __cplusplus
}
#endif

#endif /* SYNAPSWAP_H */
