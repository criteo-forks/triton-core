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

#include "dynamic_batch_scheduler.h"

#ifndef _WIN32
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#include "constants.h"
#include "server.h"
#include "triton/common/logging.h"
#include "triton/common/model_config.h"
#include "triton/common/nvtx.h"

namespace triton { namespace core {

bool
IsStaleState(Payload::State payload_state)
{
  return (
      (payload_state == Payload::State::EXECUTING) ||
      (payload_state == Payload::State::RELEASED));
}

void
FinishSkippedRequests(
    std::vector<std::deque<std::unique_ptr<InferenceRequest>>>&& requests,
    const Status& response_status, FailureReason reason)
{
  for (auto& queue : requests) {
    for (auto& request : queue) {
      InferenceRequest::RespondIfError(
          request, response_status, true /* release_requests */, reason);
    }
  }
}

void
FinishRejectedCancelledRequests(
    std::vector<std::deque<std::unique_ptr<InferenceRequest>>>&&
        rejected_requests,
    std::vector<std::deque<std::unique_ptr<InferenceRequest>>>&&
        cancelled_requests)
{
  const static Status rejected_status =
      Status(Status::Code::UNAVAILABLE, "Request timeout expired");
  const static Status cancelled_status = Status(Status::Code::CANCELLED);
  FinishSkippedRequests(
      std::move(rejected_requests), rejected_status, FailureReason::REJECTED);
  FinishSkippedRequests(
      std::move(cancelled_requests), cancelled_status, FailureReason::CANCELED);
}

// Finishes requests dropped by DrainStagingToQueue() because their priority
// level was already at its configured max_queue_size.
//
// Semantic deviation from the previous single-mutex design: max_queue_size
// used to be enforced synchronously inside Enqueue()'s critical section, so
// the frontend thread got a non-success Status back immediately and never
// took ownership of the request. Now that frontend threads hand requests off
// through 'staging_' and only the batcher thread later moves them into
// 'queue_' (where the limit is actually checked), Enqueue() has already
// returned success by the time such a request is identified. It is instead
// finished asynchronously here, the same way timed-out and cancelled
// requests are finished above.
void
FinishStagingRejectedRequests(
    std::deque<std::unique_ptr<InferenceRequest>>&& requests)
{
  for (auto& request : requests) {
    // Same message the synchronous rejection path used to produce from
    // PolicyQueue::Enqueue().
    const Status rejected_status(
        Status::Code::UNAVAILABLE,
        request->LogRequest() + "Exceeds maximum queue size");
    InferenceRequest::RespondIfError(
        request, rejected_status, true /* release_requests */,
        FailureReason::REJECTED);
  }
}

namespace {

// 'queued_batch_size_' is an advisory counter (it feeds the batcher wake
// predicate). A queue-corruption error path resets it to zero while staged
// requests it already counted may still flow into the queue, so a plain
// fetch_sub could wrap it around to a huge value; clamp at zero instead.
void
ClampedDecrement(std::atomic<size_t>& counter, const size_t count)
{
  size_t cur = counter.load(std::memory_order_relaxed);
  while (!counter.compare_exchange_weak(
      cur, cur - std::min(cur, count), std::memory_order_relaxed)) {
  }
}

}  // namespace

DynamicBatchScheduler::DynamicBatchScheduler(
    TritonModel* model, TritonModelInstance* model_instance,
    const bool dynamic_batching_enabled, const int32_t max_batch_size,
    const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
    const bool preserve_ordering,
    const std::set<int32_t>& preferred_batch_sizes,
    const uint64_t max_queue_delay_microseconds,
    const inference::ModelQueuePolicy& default_queue_policy,
    const uint64_t priority_levels, const ModelQueuePolicyMap& queue_policy_map)
    : model_(model), model_instance_(model_instance),
      model_name_(model->Name()),
      dynamic_batching_enabled_(dynamic_batching_enabled),
      queue_(default_queue_policy, priority_levels, queue_policy_map),
      stop_(false), max_batch_size_((size_t)std::max(1, max_batch_size)),
      preferred_batch_sizes_(preferred_batch_sizes),
      pending_batch_delay_ns_(max_queue_delay_microseconds * 1000),
      pending_batch_size_(0), queued_batch_size_(0),
      next_preferred_batch_size_(0),
      enforce_equal_shape_tensors_(enforce_equal_shape_tensors),
      has_optional_input_(false), preserve_ordering_(preserve_ordering)
{
  rate_limiter_ = model_->Server()->GetRateLimiter();
  // Both the server and model config should specify
  // caching enabled for model to utilize response cache.
  response_cache_enabled_ = model_->ResponseCacheEnabled() &&
                            model_->Server()->ResponseCacheEnabled();
  max_preferred_batch_size_ = 0;
  for (const auto size : preferred_batch_sizes_) {
    max_preferred_batch_size_ =
        std::max(max_preferred_batch_size_, (size_t)size);
  }

  for (const auto& input : model_->Config().input()) {
    if (input.optional()) {
      has_optional_input_ = true;
      break;
    }
  }
}

Status
DynamicBatchScheduler::Create(
    TritonModel* model, TritonModelInstance* model_instance, const int nice,
    const bool dynamic_batching_enabled, const int32_t max_batch_size,
    const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
    const bool preserve_ordering,
    const std::set<int32_t>& preferred_batch_sizes,
    const uint64_t max_queue_delay_microseconds,
    std::unique_ptr<Scheduler>* scheduler)
{
  inference::ModelDynamicBatching batcher_config;
  batcher_config.set_preserve_ordering(preserve_ordering);
  for (const auto& bs : preferred_batch_sizes) {
    batcher_config.add_preferred_batch_size(bs);
  }
  batcher_config.set_max_queue_delay_microseconds(max_queue_delay_microseconds);

  return Create(
      model, model_instance, nice, dynamic_batching_enabled, max_batch_size,
      enforce_equal_shape_tensors, batcher_config, scheduler);
}

Status
DynamicBatchScheduler::Create(
    TritonModel* model, TritonModelInstance* model_instance, const int nice,
    const bool dynamic_batching_enabled, const int32_t max_batch_size,
    const std::unordered_map<std::string, bool>& enforce_equal_shape_tensors,
    const inference::ModelDynamicBatching& batcher_config,
    std::unique_ptr<Scheduler>* scheduler)
{
  std::set<int32_t> preferred_batch_sizes;
  for (const auto size : batcher_config.preferred_batch_size()) {
    preferred_batch_sizes.insert(size);
  }

  DynamicBatchScheduler* dyna_sched = new DynamicBatchScheduler(
      model, model_instance, dynamic_batching_enabled, max_batch_size,
      enforce_equal_shape_tensors, batcher_config.preserve_ordering(),
      preferred_batch_sizes, batcher_config.max_queue_delay_microseconds(),
      batcher_config.default_queue_policy(), batcher_config.priority_levels(),
      batcher_config.priority_queue_policy());
  std::unique_ptr<DynamicBatchScheduler> sched(dyna_sched);

  sched->scheduler_thread_exit_.store(false);
  if (dynamic_batching_enabled) {
    sched->NewPayload();
    sched->scheduler_thread_ =
        std::thread([dyna_sched, nice]() { dyna_sched->BatcherThread(nice); });
  }

  scheduler->reset(sched.release());

  return Status::Success;
}

DynamicBatchScheduler::~DynamicBatchScheduler()
{
  // Signal the scheduler thread to exit and then wait for it..
  scheduler_thread_exit_.store(true);
  cv_.notify_one();
  if (scheduler_thread_.joinable()) {
    scheduler_thread_.join();
  }
}

Status
DynamicBatchScheduler::Enqueue(std::unique_ptr<InferenceRequest>& request)
{
  if (stop_) {
    return Status(
        Status::Code::UNAVAILABLE,
        request->LogRequest() +
            "Server is stopping, scheduler for model has stopped accepting new "
            "inference requests");
  }
  // If queue start timestamp hasn't been set, queue timer starts at
  // the beginning of the queueing and scheduling process. Otherwise,
  // dynamic batcher is used as component of another batcher and should not
  // overwrite the queue start timestamp.
  if (request->QueueStartNs() == 0) {
    request->CaptureQueueStartNs();
    INFER_TRACE_ACTIVITY(
        request->TraceProxy(), TRITONSERVER_TRACE_QUEUE_START,
        request->QueueStartNs());
#ifdef TRITON_ENABLE_TRACING
    request->TraceInputTensors(
        TRITONSERVER_TRACE_TENSOR_QUEUE_INPUT, "DynamicBatchScheduler Enqueue");
#endif  // TRITON_ENABLE_TRACING
  }

  // Record time at the beginning of the batcher queueing. In the case of
  // oldest sequence batcher, this will overwrite the value that was previously
  // set by sequence batcher, which is okay as by this point, the previous
  // batcher won't be needing this value and it can be safely reused by
  // the dynamic batcher.
  request->CaptureBatcherStartNs();

  std::unique_ptr<InferenceResponse> cached_response;

  if (response_cache_enabled_) {
    CacheLookUp(request, cached_response);
  }

  if (cached_response != nullptr) {
    // If there was a cache hit then try sending the cached response
    // and release the request.
    if (preserve_ordering_) {
      // In order to preserve the order, the response send must be
      // delegated.
      DelegateResponse(request);
    }

    // Send cached response and release request
    InferenceResponse::Send(
        std::move(cached_response), TRITONSERVER_RESPONSE_COMPLETE_FINAL);
    InferenceRequest::Release(
        std::move(request), TRITONSERVER_REQUEST_RELEASE_ALL);

    return Status::Success;
  }

  if (!dynamic_batching_enabled_) {
    if (preserve_ordering_ || response_cache_enabled_) {
      DelegateResponse(request);
    }
    // If not using dynamic batching, directly enqueue the
    // request to model for execution
    auto payload = model_->Server()->GetRateLimiter()->GetPayload(
        Payload::Operation::INFER_RUN, nullptr /* TritonModelInstance*/);
    payload->AddRequest(std::move(request));
    RETURN_IF_ERROR(model_->Server()->GetRateLimiter()->EnqueuePayload(
        model_, payload, CachedPayloadQueue()));

  } else {
    queued_batch_size_.fetch_add(
        std::max(1U, request->BatchSize()), std::memory_order_relaxed);

    // Hand the request off to the batcher thread through 'staging_' instead
    // of enqueuing into 'queue_' under 'mu_' here. 'queue_' is owned
    // exclusively by the batcher thread, which moves staged requests into it
    // via DrainStagingToQueue(); this keeps frontend threads from contending
    // with 'mu_' across the batcher's (potentially long) batch-formation
    // iteration. Push() cannot fail, so unlike the old queue_.Enqueue() call
    // this can no longer return a max_queue_size error synchronously (see the
    // deviation note next to FinishStagingRejectedRequests()).
    staging_.Push(std::move(request));

    // We may wake up runner less often if we don't enforce equal shape
    // within a batch, otherwise must always wake up runner to check it.
    //
    // The wake decision is re-expressed purely in terms of the advisory
    // atomics below, dropping the previous curr_payload_ exec-mutex state
    // probe: a stale answer here at worst delays the batcher wake until the
    // next enqueue or its wait timeout, and the payload's own completion
    // callback also notifies 'cv_' once it transitions to a stale state.
    bool wake_batcher = true;
    if (enforce_equal_shape_tensors_.empty()) {
      wake_batcher =
          (payload_saturated_.load(std::memory_order_relaxed) ||
           (queued_batch_size_.load(std::memory_order_relaxed) >=
            next_preferred_batch_size_.load(std::memory_order_relaxed)));
    }

    // If there are any idle runners and the queued batch size is greater or
    // equal to next preferred batch size, then wake batcher up to service
    // this request. We do the actual wake outside of the lock to avoid
    // having the woken thread immediately block on the lock.
    // Explicitly force non-blocking to prevent waiting for the slot to
    // be available.
    //
    // The slot probe is evaluated without holding 'mu_' on purpose: it
    // acquires rate-limiter internal locks that are contended by the runner
    // threads, and holding 'mu_' across it would couple frontend enqueues to
    // runner-side contention. The wake decision is advisory: a stale answer
    // at worst delays the batcher wake until the next enqueue or its wait
    // timeout.
    if (wake_batcher) {
      wake_batcher = model_->Server()->GetRateLimiter()->PayloadSlotAvailable(
          model_, model_instance_, queue_.SupportPrefetching(),
          true /*force_non_blocking*/, CachedPayloadQueue());
    }

    if (wake_batcher) {
      // Closes a lost-wakeup race against BatcherThread's use of
      // 'batcher_idle_': if the batcher is (or is about to be) parked in
      // cv_.wait_for(), this empty critical section serializes with that
      // transition so the notify below cannot run ahead of the wait. See
      // 'batcher_idle_' in the header and its use in BatcherThread.
      if (batcher_idle_.load(std::memory_order_acquire)) {
        std::lock_guard<std::mutex> lock(mu_);
      }
      cv_.notify_one();
    }
  }

  return Status::Success;
}

void
DynamicBatchScheduler::NewPayload()
{
  curr_payload_ = model_->Server()->GetRateLimiter()->GetPayload(
      Payload::Operation::INFER_RUN, model_instance_);
  payload_saturated_.store(false, std::memory_order_relaxed);
  CustomBatchInit();
}

std::deque<std::unique_ptr<InferenceRequest>>
DynamicBatchScheduler::DrainStagingToQueue()
{
  // 'mu_' mutex must be held when this function is called.
  std::deque<std::unique_ptr<InferenceRequest>> rejected;
  auto items = staging_.TakeAll();
  for (auto& request : items) {
    // Same enqueue logic the old Enqueue() critical section used.
    auto status = queue_.Enqueue(request->Priority(), request);
    if (!status.IsOk()) {
      // 'request' is still owned by us here: a non-success Status means
      // queue_.Enqueue() left ownership with the caller (see its contract).
      ClampedDecrement(
          queued_batch_size_, std::max(1U, request->BatchSize()));
      rejected.emplace_back(std::move(request));
    }
  }
  return rejected;
}

void
DynamicBatchScheduler::BatcherThread(const int nice)
{
#ifndef _WIN32
  if (setpriority(PRIO_PROCESS, syscall(SYS_gettid), nice) == 0) {
    LOG_VERBOSE(1) << "Starting dynamic-batcher thread for " << model_name_
                   << " at nice " << nice << "...";
  } else {
    LOG_VERBOSE(1) << "Starting dynamic-batcher thread for " << model_name_
                   << " at default nice (requested nice " << nice
                   << " failed)...";
  }
#else
  LOG_VERBOSE(1) << "Starting dynamic-batcher thread for " << model_name_
                 << " at default nice...";
#endif
  // For debugging/testing, delay start of threads until the queue
  // contains the specified number of entries.
  size_t delay_cnt = 0;
  {
    const char* dstr = getenv("TRITONSERVER_DELAY_SCHEDULER");
    if (dstr != nullptr) {
      delay_cnt = atoi(dstr);
      LOG_VERBOSE(1) << "Delaying batcher thread for " << model_name_
                     << " until " << delay_cnt << " queued requests...";
    }
  }

  const uint64_t default_wait_microseconds = 500 * 1000;

  while (!scheduler_thread_exit_.load()) {
    NVTX_RANGE(nvtx_, "DynamicBatcher " + model_name_);

    std::vector<std::deque<std::unique_ptr<InferenceRequest>>>
        rejected_requests, cancelled_requests;
    std::deque<std::unique_ptr<InferenceRequest>> staging_rejected_requests;
    uint64_t wait_microseconds = 0;

    // Hold the lock for as short a time as possible.
    {
      std::unique_lock<std::mutex> lock(mu_);

      // Drain 'staging_' into 'queue_' first thing, so every check below
      // that looks at 'queue_' (delay_cnt/Empty() here, and anything
      // GetDynamicBatch() does further down) sees the latest state instead
      // of whatever was last drained on a previous iteration.
      staging_rejected_requests = DrainStagingToQueue();

      {
        std::lock_guard<std::mutex> exec_lock(*(curr_payload_->GetExecMutex()));
        auto payload_state = curr_payload_->GetState();
        if (payload_saturated_.load(std::memory_order_relaxed) ||
            IsStaleState(payload_state)) {
          NewPayload();
          next_preferred_batch_size_.store(0, std::memory_order_relaxed);
        }
      }

      if (delay_cnt > 0) {
        // Debugging/testing... wait until queue contains 'delay_cnt'
        // items...
        wait_microseconds = 10 * 1000;
        if (queue_.Size() >= delay_cnt) {
          delay_cnt = 0;
        }
        LOG_VERBOSE(1) << "Delaying batcher thread " << model_name_ << " until "
                       << delay_cnt
                       << " queued requests, current total = " << queue_.Size();
      } else if (queue_.Empty()) {
        wait_microseconds = default_wait_microseconds;
      } else {
        if (payload_saturated_.load(std::memory_order_relaxed)) {
          // Finish these now: leaving the lock scope via 'continue' would
          // otherwise drop them without a response (see the equivalent
          // handling further below).
          lock.unlock();
          FinishStagingRejectedRequests(std::move(staging_rejected_requests));
          continue;
        }

        WaitForPayloadSlotAvailable(&lock, default_wait_microseconds);

        {
          std::lock_guard<std::mutex> exec_lock(
              *(curr_payload_->GetExecMutex()));

          auto payload_state = curr_payload_->GetState();
          if (IsStaleState(payload_state)) {
            lock.unlock();
            FinishStagingRejectedRequests(std::move(staging_rejected_requests));
            continue;
          }

          // Use dynamic batching to get request(s) to execute.
          wait_microseconds = GetDynamicBatch();

          // Get requests that are rejected or cancelled from searching dynamic
          // batch.
          queue_.ReleaseSkippedRequests(
              &rejected_requests, &cancelled_requests);

          // Extract batch only if there is pending batch
          auto pending_batch_queue_cnt = queue_.PendingBatchCount();
          if ((wait_microseconds == 0) && (pending_batch_queue_cnt != 0)) {
            curr_payload_->ReserveRequests(pending_batch_queue_cnt);
            for (size_t idx = 0; idx < pending_batch_queue_cnt; ++idx) {
              std::unique_ptr<InferenceRequest> request;
              auto status = queue_.Dequeue(&request);
              if (status.IsOk()) {
                if (preserve_ordering_ || response_cache_enabled_) {
                  DelegateResponse(request);
                }
                curr_payload_->AddRequest(std::move(request));
              } else {
                // The queue is empty which conflicts with pending batch
                // count. Send the current batch if any and reset related
                // variables.
                LOG_ERROR << request->LogRequest()
                          << "Failed to retrieve request from scheduler queue: "
                          << status.Message();
                queue_.ResetCursor();
                queued_batch_size_.store(0, std::memory_order_relaxed);
                pending_batch_size_ = 0;
                break;
              }
            }

            if (curr_payload_->GetState() == Payload::State::UNINITIALIZED) {
              curr_payload_->SetState(Payload::State::READY);
            }

            ClampedDecrement(queued_batch_size_, pending_batch_size_);
            pending_batch_size_ = 0;
          }
        }
      }

      // If no requests are to be handled, wait for notification or
      // for the specified timeout before checking the queue again.
      if (wait_microseconds > 0) {
        std::chrono::microseconds wait_timeout(wait_microseconds);
        // Closes a lost-wakeup race against Enqueue()'s advisory wake check
        // (see 'batcher_idle_' in the header): set while still holding
        // 'mu_', immediately before the wait releases it, and cleared right
        // after waking.
        batcher_idle_.store(true, std::memory_order_release);
        // Exact (mutex-taking) staging emptiness check: a producer may have
        // pushed between the drain at the top of this iteration and here,
        // with its notify landing before this thread parks. If its Push()
        // completed before this check we see the request and skip the wait;
        // otherwise the producer's batcher_idle_ handshake in Enqueue()
        // serializes on 'mu_' and its notify lands during the wait. Either
        // way no request can strand in staging for a full wait timeout.
        if (staging_.EmptyExact()) {
          cv_.wait_for(lock, wait_timeout);
        }
        batcher_idle_.store(false, std::memory_order_relaxed);
      }
    }

    if (curr_payload_->GetState() == Payload::State::READY) {
      auto callback = [this]() { cv_.notify_one(); };
      curr_payload_->SetCallback(callback);
      {
        std::lock_guard<std::mutex> exec_lock(*(curr_payload_->GetExecMutex()));
        CustomBatchFini();
      }
      model_->Server()->GetRateLimiter()->EnqueuePayload(
          model_, curr_payload_, CachedPayloadQueue());
    }

    // Finish rejected and cancelled requests if any
    FinishStagingRejectedRequests(std::move(staging_rejected_requests));
    FinishRejectedCancelledRequests(
        std::move(rejected_requests), std::move(cancelled_requests));
  }  // end runner loop

  LOG_VERBOSE(1) << "Stopping dynamic-batcher thread for " << model_name_
                 << "...";
}

void
DynamicBatchScheduler::WaitForPayloadSlotAvailable(
    std::unique_lock<std::mutex>* lock, uint64_t wait_microseconds)
{
  // The wait_for_slots conditional can be blocking till the slots are available
  // for execution. Need to explicitly release the 'mu_' lock to allow the
  // Enqueue threads above to make progress.
  //
  // While blocked here, request timeouts are only enforced when the wait
  // expires, so each wait window is clamped to the closest deadline of the
  // pending batch (when timeout policies are in use) instead of always
  // waiting the full 'wait_microseconds'. Without configured timeouts the
  // behavior is unchanged. Note the clamp tracks the pending batch's
  // deadlines; requests arriving during the wait are picked up when their
  // deadline enters the pending batch on a later batching pass.
  uint64_t clamped_wait_us =
      ClampWaitToClosestTimeout(wait_microseconds);  // requires 'mu_'
  lock->unlock();

  std::mutex slot_mu;
  std::unique_lock<std::mutex> slot_lock(slot_mu);
  bool slot_available = false;

  while (!slot_available) {
    slot_available = cv_.wait_for(
        slot_lock, std::chrono::microseconds(clamped_wait_us), [this]() {
          return model_->Server()->GetRateLimiter()->PayloadSlotAvailable(
              model_, model_instance_, queue_.SupportPrefetching(),
              true /* force_non_blocking */, CachedPayloadQueue());
        });
    if (!slot_available) {
      // Reject and release timeout requests from queue.
      std::vector<std::deque<std::unique_ptr<InferenceRequest>>>
          rejected_requests, cancelled_requests;
      std::deque<std::unique_ptr<InferenceRequest>> staging_rejected_requests;
      {
        std::lock_guard<std::mutex> lock(mu_);
        // Drain 'staging_' before RejectTimeoutRequests()/
        // ClampWaitToClosestTimeout() below, so requests that arrived while
        // waiting for a slot are considered for timeout rejection and the
        // next wait window right away instead of only on the next full
        // BatcherThread iteration.
        staging_rejected_requests = DrainStagingToQueue();
        queue_.RejectTimeoutRequests();
        queue_.ReleaseSkippedRequests(&rejected_requests, &cancelled_requests);
        clamped_wait_us = ClampWaitToClosestTimeout(wait_microseconds);
      }
      FinishStagingRejectedRequests(std::move(staging_rejected_requests));
      FinishRejectedCancelledRequests(
          std::move(rejected_requests), std::move(cancelled_requests));
    }
  }

  // Recapture the lock.
  lock->lock();
}

uint64_t
DynamicBatchScheduler::ClampWaitToClosestTimeout(uint64_t wait_microseconds)
{
  // 'mu_' mutex must be held when this function is called.
  const uint64_t now_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count();
  return ClampWaitToDeadline(
      wait_microseconds, queue_.ClosestTimeout(), now_ns);
}

uint64_t
DynamicBatchScheduler::ClampWaitToDeadline(
    uint64_t wait_microseconds, uint64_t closest_timeout_ns, uint64_t now_ns)
{
  if (closest_timeout_ns == 0) {
    return wait_microseconds;
  }
  if (now_ns >= closest_timeout_ns) {
    // A deadline already expired; use a minimal wait so the caller's next
    // pass rejects it promptly.
    return 1000;
  }
  // Floor at 1ms to avoid degenerating into a busy poll near a deadline.
  const uint64_t remaining_us = (closest_timeout_ns - now_ns) / 1000;
  return std::min(wait_microseconds, std::max(remaining_us, uint64_t(1000)));
}

uint64_t
DynamicBatchScheduler::GetDynamicBatch()
{
  // 'mu_' mutex must be held when this function is called. queue_
  // must not be empty.

  // Examine the new requests. If adding these new requests to the
  // pending batch allows a preferred batch size then execute it
  // immediately. Stop examining requests if the maximum preferred
  // batch size would be exceeded or if the shape of the next request
  // does not match the shape of the pending batch.
  bool send_now = false;

  // If the previous payload was not executed, reset the cursor to the start
  // of the queue to re-iterate over it and find the ideal batch.
  if (!queue_.IsCursorValid()) {
    queue_.ResetCursor();
    pending_batch_size_ = 0;
    if (CustomBatchEnabled()) {
      CustomBatchFini();
      CustomBatchInit();
    }
  }
  size_t best_preferred_batch_size = 0;
  ClampedDecrement(queued_batch_size_, queue_.ApplyPolicyAtCursor());

  // When there is optional input or input shape must be enforced,
  // the inputs in the requests must be examined for forming a batch
  const bool check_input =
      !enforce_equal_shape_tensors_.empty() || has_optional_input_;
  auto payload_batch_size = curr_payload_->BatchSize();
  while (!queue_.CursorEnd()) {
    const auto batch_size = std::max(1U, queue_.RequestAtCursor()->BatchSize());

    // If there is no pending batch, then this request is starting a
    // new batch.
    if ((payload_batch_size + queue_.PendingBatchCount()) == 0) {
      // Get the shape of the new batch that is being started...
      if (check_input) {
        if (!curr_payload_->MutableRequiredEqualInputs()
                 ->Initialize(
                     queue_.RequestAtCursor(), enforce_equal_shape_tensors_,
                     has_optional_input_)
                 .IsOk()) {
          send_now = true;
          break;
        }
      }
    } else {
      // There is a pending batch and adding this request would make
      // the batch size larger than all of the preferred batch sizes,
      // so mark the cursor at this point. Not sending the pending batch so
      // that we can examine the queue delay of requests that fits in a batch.
      if (((payload_batch_size + pending_batch_size_ + batch_size) >
           max_preferred_batch_size_) &&
          (best_preferred_batch_size == 0)) {
        best_preferred_batch_size = pending_batch_size_;
        queue_.MarkCursor();
        payload_saturated_.store(true, std::memory_order_relaxed);
      }
      if ((payload_batch_size + pending_batch_size_ + batch_size) >
          max_batch_size_) {
        send_now = true;
        break;
      }

      // There is a pending batch and it has a different shape then
      // this request, so send the pending batch as it is.
      if (check_input &&
          !curr_payload_->MutableRequiredEqualInputs()->HasEqualInputs(
              queue_.RequestAtCursor())) {
        curr_payload_->MarkSaturated();
        send_now = true;
        break;
      }
    }

    if (CustomBatchEnabled()) {
      bool should_include = false;
      CustomBatchIncl(queue_.RequestAtCursor().get(), &should_include);
      if (!should_include) {
        curr_payload_->MarkSaturated();
        send_now = true;
        break;
      }
    }

    pending_batch_size_ += batch_size;
    queue_.AdvanceCursor();
    ClampedDecrement(queued_batch_size_, queue_.ApplyPolicyAtCursor());

    if (preferred_batch_sizes_.find(pending_batch_size_ + payload_batch_size) !=
        preferred_batch_sizes_.end()) {
      best_preferred_batch_size = pending_batch_size_;
      queue_.MarkCursor();
    }
  }

  // Obtain the age of the oldest pending request to compare with the maximum
  // batch queuing delay
  uint64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
  uint64_t delay_ns = now_ns - queue_.OldestEnqueueTime();
  bool delay_is_exceeded =
      (pending_batch_delay_ns_ != 0) && (delay_ns >= pending_batch_delay_ns_);

  // If we found a preferred batch size and the queue delay hasn't
  // been exceeded, then execute that.
  if ((best_preferred_batch_size != 0) && !delay_is_exceeded) {
    if (pending_batch_delay_ns_ == 0) {
      payload_saturated_.store(true, std::memory_order_relaxed);
    }
    pending_batch_size_ = best_preferred_batch_size;
    queue_.SetCursorToMark();
    return 0;
  }

  // No request in pending batch happens when all queued requests have expired
  // timeout and the policies are REJECT
  if (queue_.PendingBatchCount() == 0) {
    return 0;
  }

  // If the delay has been exceeded, or if the current batch can't grow
  // any larger then just immediately execute whatever is pending.
  if (send_now || ((payload_batch_size + pending_batch_size_) >=
                   max_preferred_batch_size_)) {
    payload_saturated_.store(true, std::memory_order_relaxed);
    return 0;
  }

  if (delay_is_exceeded || (pending_batch_delay_ns_ == 0)) {
    return 0;
  }

  // Set the next preferred batch size given the pending batch size
  auto next_preferred_batch_size_it = preferred_batch_sizes_.upper_bound(
      pending_batch_size_ + payload_batch_size);
  if (next_preferred_batch_size_it != preferred_batch_sizes_.end()) {
    next_preferred_batch_size_.store(
        *next_preferred_batch_size_it, std::memory_order_relaxed);
  } else {
    next_preferred_batch_size_.store(
        preferred_batch_sizes_.empty() ? 0 : *preferred_batch_sizes_.begin(),
        std::memory_order_relaxed);
  }
  if (next_preferred_batch_size_.load(std::memory_order_relaxed) != 0) {
    // GetDynamicBatch() runs exclusively on the batcher thread, so this
    // load-then-store is race-free with itself; concurrent readers
    // (Enqueue()'s wake check) only ever see this as one advisory value or
    // the other.
    next_preferred_batch_size_.fetch_sub(
        payload_batch_size, std::memory_order_relaxed);
  }

  // By this point, we have not seen the pending batch that should be executed
  // immediately. However, if we have scheduled a payload that can be grown and
  // not yet in preferred batch size, we should move the pending batch over to
  // ensure the model instance will pick up largest available batch even if it
  // is not the preferred batch.
  if (!payload_saturated_.load(std::memory_order_relaxed) &&
      (payload_batch_size != 0) &&
      (preferred_batch_sizes_.find(payload_batch_size) ==
       preferred_batch_sizes_.end())) {
    return 0;
  }

  uint64_t wait_ns = pending_batch_delay_ns_ - delay_ns;
  // Note that taking request timeout into consideration allows us to reset
  // pending batch as soon as it is invalidated. But the cost is that in edge
  // case where the timeout will be expired one by one, the thread will be
  // waken frequently.
  if (queue_.ClosestTimeout() != 0) {
    if (now_ns <= queue_.ClosestTimeout()) {
      wait_ns = std::min(queue_.ClosestTimeout() - now_ns, wait_ns);
    } else {
      // A request in pending batch is timed-out, wait for 1 us to force the
      // thread to reset the pending batch right the way.
      wait_ns = 1000;
    }
  }

  // Return non-zero wait microseconds to cause this thread to wait
  // until the queue delay or the closest timeout has expired.
  // Another thread may be awaken due to incoming request to handle the
  // pending batch before this thread wakes and that is ok. But if no other
  // request comes in then this thread will wake and revisit the pending batch
  // (and at that time will then see the delay has been exceeded and will send
  // the batch).
  return wait_ns / 1000;
}

void
DynamicBatchScheduler::DelegateResponse(
    std::unique_ptr<InferenceRequest>& request)
{
  // Cache plumbing
  const std::string& key = request->CacheKey();
  const bool is_key_set = request->CacheKeyIsSet();
  const uint64_t lookup_end_ns = request->CacheLookupEndNs();
  const uint64_t lookup_start_ns = request->CacheLookupStartNs();

  request->SetResponseDelegator(
      [this, key, is_key_set, lookup_end_ns, lookup_start_ns](
          std::unique_ptr<InferenceResponse>&& response, const uint32_t flags) {
        if (response_cache_enabled_) {
          // Logical error, the key should be set if caching is enabled
          // for this model
          if (!is_key_set) {
            LOG_ERROR << "Request cache key was not set correctly.";
          }

          // Cache insertion happens here because we need the backend to have
          // computed the inference response first in the case of cache miss
          auto cache = model_->Server()->CacheManager()->Cache();

#ifdef TRITON_ENABLE_STATS
          const uint64_t insert_start_ns = CaptureTimeNs();
#endif  // TRITON_ENABLE_STATS

          auto status = cache->Insert(response.get(), key);

#ifdef TRITON_ENABLE_STATS
          const uint64_t insert_end_ns = CaptureTimeNs();
#endif  // TRITON_ENABLE_STATS

          bool cache_miss =
              (status.StatusCode() != Status::Code::ALREADY_EXISTS);
          if (cache_miss) {
#ifdef TRITON_ENABLE_STATS
            uint64_t lookup_ns = lookup_end_ns - lookup_start_ns;
            // Logical error, this shouldn't happen
            if (lookup_start_ns > lookup_end_ns) {
              lookup_ns = 0;
              LOG_ERROR << "Request lookup duration was not set correctly.";
            }

            uint64_t insert_ns = insert_end_ns - insert_start_ns;
            uint64_t cache_miss_ns = lookup_ns + insert_ns;
            // Use model_ to update stats directly because request object can be
            // released by the backend before getting to this callback.
            model_->MutableStatsAggregator()->UpdateSuccessCacheMiss(
                model_->MetricReporter().get(), cache_miss_ns);
#endif  // TRITON_ENABLE_STATS
            if (!status.IsOk()) {
              LOG_ERROR << "Failed to insert key [" << key
                        << "] into response cache: " << status.Message();
            }
          }  // Otherwise do nothing; we update cache hit statistics on Lookup
        }

        if (preserve_ordering_) {
          {
            std::lock_guard<std::mutex> lock(completion_queue_mtx_);
            completion_queue_.emplace_back();
            auto queue_slot = &completion_queue_.back();
            queue_slot->emplace_back(std::move(response), flags);
          }
          FinalizeResponses();
        } else {
          InferenceResponse::Send(std::move(response), flags);
        }
      });
}

void
DynamicBatchScheduler::CacheLookUp(
    std::unique_ptr<InferenceRequest>& request,
    std::unique_ptr<InferenceResponse>& cached_response)
{
  auto cache = model_->Server()->CacheManager()->Cache();
  bool is_lookup_success = CacheLookUpUtil(request, cached_response, cache);
  if (is_lookup_success) {
#ifdef TRITON_ENABLE_STATS
    // Update model metrics/stats on cache hits
    // Backends will update metrics as normal on cache misses
    request->ReportStatisticsCacheHit(model_->MetricReporter().get());
#endif  // TRITON_ENABLE_STATS
  }
}

void
DynamicBatchScheduler::FinalizeResponses()
{
  // Need exclusive access of the function to ensure responses are sent
  // in order
  std::lock_guard<std::mutex> lock(finalize_mtx_);
  // Finalize the completed payloads in-order as far as possible
  std::deque<std::pair<std::unique_ptr<InferenceResponse>, const uint32_t>>
      responses;
  {
    std::lock_guard<std::mutex> queue_lock(completion_queue_mtx_);
    while (!completion_queue_.empty() && !completion_queue_.front().empty()) {
      bool response_complete = false;
      for (auto& response_pair : completion_queue_.front()) {
        // Assuming FINAL flag is set only in the last response of the request
        response_complete =
            ((response_pair.second & TRITONSERVER_RESPONSE_COMPLETE_FINAL) !=
             0);
        responses.emplace_back(std::move(response_pair));
      }
      if (response_complete) {
        completion_queue_.pop_front();
      } else {
        completion_queue_.front().clear();
      }
    }
  }

  for (auto& response : responses) {
    InferenceResponse::Send(std::move(response.first), response.second);
  }
}

bool
DynamicBatchScheduler::CustomBatchEnabled() const
{
  return model_->ModelBatchInitFn();
}

void
DynamicBatchScheduler::CustomBatchIncl(
    InferenceRequest* request, bool* should_include)
{
  if (!CustomBatchEnabled())
    return;
  TRITONSERVER_Error* err = model_->ModelBatchInclFn()(
      reinterpret_cast<TRITONBACKEND_Request*>(request),
      *curr_payload_.get()->UserPointerAddr(), should_include);
  if (err) {
    LOG_ERROR << "Custom batching include function failed for model "
              << model_->Name() << ": " << TRITONSERVER_ErrorMessage(err);
    TRITONSERVER_ErrorDelete(err);
  }
}

void
DynamicBatchScheduler::CustomBatchInit()
{
  if (!CustomBatchEnabled())
    return;
  TRITONSERVER_Error* err = model_->ModelBatchInitFn()(
      *model_->Batcher(), curr_payload_.get()->UserPointerAddr());
  if (err != nullptr) {
    LOG_ERROR << "Custom batching initialization function failed for model "
              << model_->Name() << ": " << TRITONSERVER_ErrorMessage(err);
    TRITONSERVER_ErrorDelete(err);
  }
}

void
DynamicBatchScheduler::CustomBatchFini()
{
  if (!CustomBatchEnabled() ||
      *curr_payload_.get()->UserPointerAddr() == nullptr)
    return;
  TRITONSERVER_Error* err =
      model_->ModelBatchFiniFn()(*curr_payload_.get()->UserPointerAddr());
  *curr_payload_.get()->UserPointerAddr() = nullptr;

  if (err != nullptr) {
    LOG_ERROR << "Custom batching finalization function failed for model "
              << model_->Name() << ": " << TRITONSERVER_ErrorMessage(err);
    TRITONSERVER_ErrorDelete(err);
  }
}

}}  // namespace triton::core
