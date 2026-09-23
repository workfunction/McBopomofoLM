// slothe.h — reusable core API for the slothe ggml forward pass.
#pragma once
#include <cstddef>
#include <cstdint>

struct slothe_model;

slothe_model * slothe_load(const char * gguf_path);
void           slothe_free(slothe_model * m);
int            slothe_n_char(const slothe_model * m);
// McBopomofoLM patch 2: slothe_load returns nullptr on any bad file (never exit()s).
int            slothe_n_syl(const slothe_model * m);
// out_logits: caller-owned buffer of T * n_char floats, row-major [T][n_char].
void           slothe_logits(slothe_model * m, const int32_t * syl_ids, int T, float * out_logits);
// McBopomofoLM patch: per-length compute-graph cache (LRU) controls and stats.
void           slothe_set_graph_cache_capacity(slothe_model * m, int capacity);
void           slothe_graph_cache_stats(const slothe_model * m, int * n_graphs, size_t * compute_bytes, uint64_t * builds);
