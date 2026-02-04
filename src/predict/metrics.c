/**
 * @file metrics.c
 * @brief Real-time performance analysis and dashboard.
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include "internal_alloc.h"
#include "synapswap.h"

extern ss_context_t g_ctx;

/* --- High-Precision Timing Utilities --- */

static double _get_time_ms() {
#ifdef _WIN32
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / frequency.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * 1000.0) + (ts.tv_nsec / 1000000.0);
#endif
}

static double g_stall_start_time = 0;
static double g_accumulated_stall_ms = 0;

/**
 * @brief Records a memory access and updates the prefetch hit rate using EMA.
 */
void metrics_record_access(bool hit, bool was_prefetch) {
    // Suppress unused parameter warning
    (void)was_prefetch;

    pthread_mutex_lock(&g_ctx.stats_mutex);
    
    const float alpha = 0.05f;
    float current_val = hit ? 1.0f : 0.0f;
    
    // First pass: initialize instead of averaging
    if (g_ctx.stats.total_access_count == 0) {
        g_ctx.stats.prefetch_hit_rate = current_val;
    } else {
        g_ctx.stats.prefetch_hit_rate = (alpha * current_val) + (1.0f - alpha) * g_ctx.stats.prefetch_hit_rate;
    }
    
    g_ctx.stats.total_access_count++;
    if (!hit) {
        g_ctx.stats.total_transfers++;
    }
    
    pthread_mutex_unlock(&g_ctx.stats_mutex);
}

void metrics_start_stall() {
    g_stall_start_time = _get_time_ms();
}

void metrics_end_stall() {
    if (g_stall_start_time > 0) {
        double delta = _get_time_ms() - g_stall_start_time;
        g_accumulated_stall_ms += delta;
        g_stall_start_time = 0;
    }
}

/**
 * @brief Public monitoring interface.
 */
SS_API void synapswap_get_stats(ss_stats_t* stats) {
    if (!stats) return;
    pthread_mutex_lock(&g_ctx.stats_mutex);
    
    memcpy(stats, &g_ctx.stats, sizeof(ss_stats_t));
    stats->vram_total_bytes = g_ctx.vram_limit;
    stats->vram_used_bytes = g_ctx.vram_allocated;
    stats->avg_transfer_latency_ms = g_accumulated_stall_ms; 
    
    pthread_mutex_unlock(&g_ctx.stats_mutex);
}

/**
 * @brief Displays a visual dashboard in the console.
 */
void metrics_display_current() {
    ss_stats_t s;
    synapswap_get_stats(&s);

    double vram_mb = (double)s.vram_used_bytes / (1024.0 * 1024.0);
    double limit_mb = (double)s.vram_total_bytes / (1024.0 * 1024.0);
    double usage_pct = (limit_mb > 0) ? (vram_mb / limit_mb) * 100.0 : 0;

    // Build the usage bar
    char bar[21];
    int progress = (int)(usage_pct / 5.0);
    for (int i = 0; i < 20; i++) {
        bar[i] = (i < progress) ? '|' : ' ';
    }
    bar[20] = '\0';

    printf("\n\033[1;36m[SynapSwap Dashboard]\033[0m\n");
    printf(" ├─ VRAM:   [%s] %4.1f%% (%.1f/%.1f MB)\n", 
           bar, usage_pct, vram_mb, limit_mb);
    
    // Hit Rate color: green if > 80%, yellow otherwise
    const char* hr_color = (s.prefetch_hit_rate > 0.8f) ? "\033[1;32m" : "\033[1;33m";
    printf(" ├─ Hit Rate: %s%5.1f%%\033[0m\n", hr_color, s.prefetch_hit_rate * 100.0);
    
    printf(" ├─ Swaps:    %lu\n", (unsigned long)s.total_transfers);
    printf(" └─ Stall:    \033[1;31m%.2f ms\033[0m\n", g_accumulated_stall_ms);
}
