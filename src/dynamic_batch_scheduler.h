// Copyright 2018-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <map>
#include <mutex>
#include <queue>
#include <set>
#include <thread>

#include "backend_model.h"
#include "backend_model_instance.h"
#include "model_config.pb.h"
#include "rate_limiter.h"
#include "scheduler.h"
#include "scheduler_utils.h"
#include "staging_queue.h"
#include "status.h"
#include "triton/common/model_config.h"

namespace triton { namespace core {

// Scheduler that implements dynamic batching.
class DynamicBatchScheduler : public Scheduler {
 public:
  // Create a scheduler to support a given number of runners and a run
  // function to call when a request is scheduled.
  static Status Create(
      TritonModel* model, TritonModelInstance* model_instance, const int nice,
      const bool dynamic_batching_enabled, const int32_t max_batch_size,
      const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
      const bool preserve_ordering,
      const std::set<int32_t>& preferred_batch_sizes,
      const uint64_t max_queue_delay_microseconds,
      std::unique_ptr<Scheduler>* scheduler);

  // Create a scheduler to support a given number of runners and a run
  // function to call when a request is scheduled. And the scheduler also
  // supports different queue policies for different priority levels.
  static Status Create(
      TritonModel* model, TritonModelInstance* model_instance, const int nice,
      const bool dynamic_batching_enabled, const int32_t max_batch_size,
      const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
      const inference::ModelDynamicBatching& batcher_config,
      std::unique_ptr<Scheduler>* scheduler);

  ~DynamicBatchScheduler();

  // \see Scheduler::Enqueue()
  Status Enqueue(std::unique_ptr<InferenceRequest>& request) override;

  // \see Scheduler::InflightInferenceCount()
  size_t InflightInferenceCount() override
  {
    std::unique_lock<std::mutex> lock(mu_);
    // 'staging_' holds requests handed off by frontend threads that the
    // batcher hasn't yet drained into 'queue_'; its size is best-effort
    // (see StagingQueue::Size()), which is acceptable for this diagnostic
    // count.
    size_t count = queue_.Size() + staging_.Size();
    if (curr_payload_ != nullptr) {
      count += curr_payload_->RequestCount();
    }
    return count;
  }

  // \see Scheduler::Stop()
  void Stop() override { stop_ = true; }

 private:
  DynamicBatchScheduler(
      TritonModel* model, TritonModelInstance* model_instance,
      const bool dynamic_batching_enabled, const int32_t max_batch_size,
      const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
      const bool preserve_ordering,
      const std::set<int32_t>& preferred_batch_sizes,
      const uint64_t max_queue_delay_microseconds,
      const inference::ModelQueuePolicy& default_queue_policy,
      const uint64_t priority_levels,
      const ModelQueuePolicyMap& queue_policy_map);

  void BatcherThread(const int nice);
  void NewPayload();
  uint64_t GetDynamicBatch();

  // Drains all requests staged via 'staging_' into 'queue_', applying the
  // same per-priority enqueue logic the old single-mutex Enqueue() critical
  // section used. 'mu_' must be held when this function is called. Returns
  // any requests that could not be enqueued because their priority level was
  // already at its configured max_queue_size (see the max_queue_size
  // deviation note next to FinishStagingRejectedRequests() in the .cc file).
  std::deque<std::unique_ptr<InferenceRequest>> DrainStagingToQueue();
  void DelegateResponse(std::unique_ptr<InferenceRequest>& request);
  void CacheLookUp(
      std::unique_ptr<InferenceRequest>& request,
      std::unique_ptr<InferenceResponse>& cached_response);
  void FinalizeResponses();

  // Block until a payload slot is available on the rate limiter. The 'lock'
  // should be acquired when calling this function. The 'lock' will be released
  // when waiting for payload slot and re-acquired before this function returns.
  // For queued requests under policy REJECT, they will be rejected if timed-out
  // while waiting for a slot. The timeout will be checked every
  // 'wait_microseconds', or sooner when the pending batch holds a closer
  // request deadline. The 'wait_microseconds' should be non-zero.
  void WaitForPayloadSlotAvailable(
      std::unique_lock<std::mutex>* lock, uint64_t wait_microseconds);

  // Returns 'wait_microseconds' clamped to the time remaining until the
  // closest request deadline in the pending batch (1ms floor), or unchanged
  // when no timeout policy is in effect. 'mu_' must be held.
  uint64_t ClampWaitToClosestTimeout(uint64_t wait_microseconds);

 public:
  // Pure deadline math backing ClampWaitToClosestTimeout(); public and
  // static for unit testing. 'closest_timeout_ns' == 0 means no deadline.
  static uint64_t ClampWaitToDeadline(
      uint64_t wait_microseconds, uint64_t closest_timeout_ns,
      uint64_t now_ns);

 private:
  // Custom batching function calls
  // Returns whether custom batching is enabled.
  bool CustomBatchEnabled() const;
  // If custom batching is enabled for this model, see if this request should be
  // included.
  void CustomBatchIncl(InferenceRequest* request, bool* should_include);
  // If custom batching is enabled for this model, initialize the batching
  // function.
  void CustomBatchInit();
  // If custom batching is enabled for this model, finalizethe batching
  // function.
  void CustomBatchFini();

  TritonModel* model_;
  TritonModelInstance* model_instance_;

  // Name of the model.
  std::string model_name_;

  // True if dynamic batching is enabled.
  const bool dynamic_batching_enabled_;

  // Map from priority level to queue holding inference requests for the model
  // represented by this scheduler. If priority queues are not supported by the
  // scheduler, then priority zero entry is used as the single queue.
  // Owned exclusively by the batcher thread; frontend threads never touch it
  // directly, they hand requests off through 'staging_' instead (see
  // DrainStagingToQueue()).
  PriorityQueue queue_;
  bool stop_;

  // Small hand-off queue frontend (Enqueue()) threads push into, with its
  // own tiny lock. The batcher thread drains it into 'queue_' instead of
  // enqueuing directly, so frontend threads no longer contend with 'mu_'
  // across the batcher's (potentially long) batch-formation iteration.
  StagingQueue<InferenceRequest> staging_;

  std::thread scheduler_thread_;
  std::atomic<bool> scheduler_thread_exit_;

  // Mutex and condvar for signaling scheduler thread
  std::mutex mu_;
  std::condition_variable cv_;

  // Set (while holding 'mu_') immediately before the batcher thread parks in
  // cv_.wait_for() and cleared right after it wakes. Enqueue() uses it to
  // close a lost-wakeup race: when a producer's advisory wake check sees
  // this true, it takes an empty lock on 'mu_' before notifying, which
  // serializes it with the batcher's wait/wake transition (see BatcherThread
  // and Enqueue() in the .cc file for the two halves of this handshake).
  std::atomic<bool> batcher_idle_{false};

  std::shared_ptr<RateLimiter> rate_limiter_;

  // Cached payload queue handle for 'model_', lazily resolved. The handle
  // stays valid until the model is unregistered from the rate limiter, which
  // outlives this scheduler. Atomic because Enqueue() runs on many frontend
  // threads concurrently (the racing writes all store the same value).
  std::atomic<RateLimiter::PayloadQueue*> payload_queue_{nullptr};

  // Returns the cached payload queue handle, resolving it on first use.
  // May return nullptr if no instance has been registered yet.
  RateLimiter::PayloadQueue* CachedPayloadQueue()
  {
    RateLimiter::PayloadQueue* pq =
        payload_queue_.load(std::memory_order_relaxed);
    if (pq == nullptr) {
      pq = rate_limiter_->LookupPayloadQueue(model_);
      if (pq != nullptr) {
        payload_queue_.store(pq, std::memory_order_relaxed);
      }
    }
    return pq;
  }

  std::shared_ptr<Payload> curr_payload_;
  // Read by Enqueue() (frontend threads) as part of the advisory wake
  // decision, written by the batcher thread (NewPayload(), GetDynamicBatch());
  // relaxed ordering is sufficient since a stale read only delays a wake.
  std::atomic<bool> payload_saturated_;

  size_t max_batch_size_;
  size_t max_preferred_batch_size_;
  std::set<int32_t> preferred_batch_sizes_;
  uint64_t pending_batch_delay_ns_;
  size_t pending_batch_size_;

  // Read by Enqueue() (frontend threads) as part of the advisory wake
  // decision, updated by frontend threads (fetch_add) and by the batcher
  // thread (GetDynamicBatch()); relaxed ordering is sufficient, see
  // StagingQueue's own comment for the same tradeoff.
  std::atomic<size_t> queued_batch_size_;
  std::atomic<size_t> next_preferred_batch_size_;

  // The input tensors that require shape checking before being
  // allowed in a batch. As a map from the tensor name to a bool. If
  // tensor is in map then its shape must match shape of same tensor
  // in requests already in the batch. If value is "true" then
  // additional tensor is treated as a shape tensor and the values
  // contained in the shape tensor must match same tensor already in
  // the batch.
  const std::unordered_map<std::string, bool> enforce_equal_shape_tensors_;

  // Store information on whether the model contains optional inputs.
  bool has_optional_input_;

  // If true the ordering of responses matches the order of requests
  // even when there are multiple scheduler threads.
  const bool preserve_ordering_;

  // If true, the scheduler will try to retrieve responses from cache.
  bool response_cache_enabled_;

  // Per completion-id queues to store the ready responses
  std::deque<
      std::vector<std::pair<std::unique_ptr<InferenceResponse>, uint32_t>>>
      completion_queue_;
  // Lock to protect the completion_queues_
  std::mutex completion_queue_mtx_;

  // Preserves the order in which responses are finalized
  std::mutex finalize_mtx_;
};

}}  // namespace triton::core
