// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifdef TRITON_INSTANCE_QUEUE_TEST_BACKEND
#include "triton/core/tritonbackend.h"

extern "C" TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance*, TRITONBACKEND_Request**, const uint32_t)
{
  return nullptr;
}
#else

#include <chrono>
#include <filesystem>
#include <future>
#include <memory>
#include <thread>
#include <vector>

#include "dynamic_batch_scheduler.h"
#include "gtest/gtest.h"
#include "instance_queue.h"
#include "payload.h"
#include "server.h"

namespace triton { namespace core {

namespace {

// Builds an empty INFER_RUN payload. Empty payloads are sufficient here: they
// carry no requests, so they always satisfy the batch-size condition and merge
// unconditionally once the queue delay has elapsed (BatcherStartNs() == 0).
std::shared_ptr<Payload>
MakeInferPayload()
{
  auto payload = std::make_shared<Payload>();
  payload->Reset(Payload::Operation::INFER_RUN, nullptr /* instance */);
  return payload;
}

// Reproduces the waiting-consumer bookkeeping that RateLimiter performs around
// a single per-model InstanceQueue and asserts it does not drift when Dequeue
// merges payloads.
//
// RateLimiter drives the counter with exactly two moves on this queue:
//   * EnqueuePayload  -> DecrementConsumerCount(), once per enqueued payload.
//   * DequeuePayload  -> IncrementConsumerCount(), once per dequeue call.
// When Dequeue merges k payloads into one, the k payloads were each decremented
// at enqueue but only a single increment is issued for the dequeue call, so the
// count leaks -(k-1) per merge. Over many merge-heavy rounds this drives
// waiting_consumer_count_ negative, which throttles the dynamic batcher onto
// its slow 500 ms poll fallback (the dispatch gates require
// WaitingConsumerCount() > 0), degrading throughput.
//
// With the fix (Dequeue credits back one count per merged payload) the counter
// returns to the idle-instance count after every round.
TEST(InstanceQueueTest, ConsumerCountStableAcrossMerges)
{
  constexpr size_t kMaxBatchSize = 8;
  // Small, non-zero delay: pending payloads are always "old enough" to merge.
  constexpr uint64_t kMaxQueueDelayNs = 1000;
  constexpr int kNumInstances = 4;
  constexpr int kRounds = 50;
  // 1 primary payload + (kBurst - 1) merged payloads per dequeue.
  constexpr int kBurst = 3;

  InstanceQueue queue(kMaxBatchSize, kMaxQueueDelayNs);

  // Model start-up: every instance parks in DequeuePayload once, announcing
  // itself as a waiting consumer.
  for (int i = 0; i < kNumInstances; ++i) {
    queue.IncrementConsumerCount();
  }
  ASSERT_EQ(queue.WaitingConsumerCount(), kNumInstances);

  for (int round = 0; round < kRounds; ++round) {
    // Producer enqueues a burst; each enqueue claims one waiting consumer.
    for (int b = 0; b < kBurst; ++b) {
      auto payload = MakeInferPayload();
      queue.Enqueue(payload);
      queue.DecrementConsumerCount();
    }

    // One consumer dequeues the whole burst as a single merged batch.
    std::shared_ptr<Payload> payload;
    std::vector<std::shared_ptr<Payload>> merged_payloads;
    queue.Dequeue(&payload, &merged_payloads);

    ASSERT_NE(payload, nullptr);
    ASSERT_EQ(merged_payloads.size(), static_cast<size_t>(kBurst - 1))
        << "test setup expects all burst payloads to merge into one dequeue";

    // Consumer finishes and re-parks for the next round.
    queue.IncrementConsumerCount();
  }

  // Once all payloads have been consumed and every consumer is idle again, the
  // waiting-consumer count must equal the true idle-instance count. Without the
  // fix it ends deeply negative (kNumInstances - kRounds * (kBurst - 1)).
  EXPECT_EQ(queue.WaitingConsumerCount(), kNumInstances)
      << "waiting_consumer_count_ drifted across merges; the dynamic batcher "
         "would eventually be throttled onto its slow poll fallback";
}

// A dequeue that merges nothing (single queued payload) must not change the
// waiting-consumer count beyond the single increment the caller issues.
TEST(InstanceQueueTest, ConsumerCountUnchangedWithoutMerge)
{
  constexpr size_t kMaxBatchSize = 8;
  constexpr uint64_t kMaxQueueDelayNs = 1000;

  InstanceQueue queue(kMaxBatchSize, kMaxQueueDelayNs);
  queue.IncrementConsumerCount();  // one idle instance
  ASSERT_EQ(queue.WaitingConsumerCount(), 1);

  auto payload_in = MakeInferPayload();
  queue.Enqueue(payload_in);
  queue.DecrementConsumerCount();

  std::shared_ptr<Payload> payload;
  std::vector<std::shared_ptr<Payload>> merged_payloads;
  queue.Dequeue(&payload, &merged_payloads);

  ASSERT_NE(payload, nullptr);
  EXPECT_TRUE(merged_payloads.empty());

  queue.IncrementConsumerCount();  // consumer re-parks
  EXPECT_EQ(queue.WaitingConsumerCount(), 1);
}

TEST(InstanceQueueTest, TimedWaitSeesConsumerBeforeWaiting)
{
  InstanceQueue queue(1, 0);
  queue.IncrementConsumerCount();
  EXPECT_TRUE(queue.WaitForConsumer(std::chrono::microseconds(0)));
  // Waiting observes availability without reserving the consumer.
  EXPECT_EQ(queue.WaitingConsumerCount(), 1);
}

TEST(InstanceQueueTest, TimedWaitExpiresWithoutConsumer)
{
  InstanceQueue queue(1, 0);
  EXPECT_FALSE(queue.WaitForConsumer(std::chrono::milliseconds(10)));
  EXPECT_EQ(queue.WaitingConsumerCount(), 0);
}

TEST(InstanceQueueTest, TimedWaitWakesWhenConsumerBecomesAvailable)
{
  InstanceQueue queue(1, 0);
  auto waiter = std::async(std::launch::async, [&queue]() {
    return queue.WaitForConsumer(std::chrono::seconds(2));
  });
  EXPECT_EQ(
      waiter.wait_for(std::chrono::milliseconds(50)),
      std::future_status::timeout);
  queue.IncrementConsumerCount();
  EXPECT_EQ(
      waiter.wait_for(std::chrono::milliseconds(250)),
      std::future_status::ready);
  EXPECT_TRUE(waiter.get());
}

TEST(InstanceQueueTest, TimedWaitRequiresPositiveConsumerCount)
{
  InstanceQueue queue(1, 0);
  queue.DecrementConsumerCount();
  auto waiter = std::async(std::launch::async, [&queue]() {
    return queue.WaitForConsumer(std::chrono::seconds(2));
  });
  queue.IncrementConsumerCount();
  EXPECT_EQ(
      waiter.wait_for(std::chrono::milliseconds(50)),
      std::future_status::timeout);
  queue.IncrementConsumerCount();
  EXPECT_EQ(
      waiter.wait_for(std::chrono::milliseconds(250)),
      std::future_status::ready);
  EXPECT_TRUE(waiter.get());
}

TEST(InstanceQueueTest, ConsumerNotificationRacingWithTimedWaitIsNotLost)
{
  for (int i = 0; i < 100; ++i) {
    InstanceQueue queue(1, 0);
    std::promise<void> start;
    auto waiter = std::async(std::launch::async, [&queue, &start]() {
      start.set_value();
      return queue.WaitForConsumer(std::chrono::seconds(2));
    });
    start.get_future().wait();
    queue.IncrementConsumerCount();
    EXPECT_EQ(
        waiter.wait_for(std::chrono::milliseconds(250)),
        std::future_status::ready)
        << "iteration " << i;
    EXPECT_TRUE(waiter.get());
  }
}

// SizeApprox() mirrors the queue depth without taking the queue lock; it
// backs the advisory payload prefetch-cap probe. It must track Enqueue and
// plain Dequeue exactly at quiescence.
TEST(InstanceQueueTest, SizeApproxTracksEnqueueDequeue)
{
  constexpr size_t kMaxBatchSize = 8;
  // Zero delay: payload merging disabled, dequeues pop exactly one payload.
  InstanceQueue queue(kMaxBatchSize, 0 /* max_queue_delay_ns */);

  EXPECT_EQ(queue.SizeApprox(), 0u);

  constexpr int kCount = 5;
  for (int i = 0; i < kCount; ++i) {
    queue.Enqueue(MakeInferPayload());
    EXPECT_EQ(queue.SizeApprox(), static_cast<size_t>(i + 1));
    EXPECT_EQ(queue.SizeApprox(), queue.Size());
  }

  for (int i = kCount; i > 0; --i) {
    std::shared_ptr<Payload> payload;
    std::vector<std::shared_ptr<Payload> > merged_payloads;
    queue.Dequeue(&payload, &merged_payloads);
    EXPECT_TRUE(merged_payloads.empty());
    EXPECT_EQ(queue.SizeApprox(), static_cast<size_t>(i - 1));
    EXPECT_EQ(queue.SizeApprox(), queue.Size());
  }
}

// A merging dequeue removes the primary payload plus every merged payload;
// SizeApprox() must account for all of them, or the prefetch cap would
// permanently believe the queue is fuller than it is and throttle the
// batcher.
TEST(InstanceQueueTest, SizeApproxTracksMergedDequeues)
{
  constexpr size_t kMaxBatchSize = 8;
  constexpr uint64_t kMaxQueueDelayNs = 1000;  // merging enabled
  constexpr int kBurst = 3;

  InstanceQueue queue(kMaxBatchSize, kMaxQueueDelayNs);

  for (int b = 0; b < kBurst; ++b) {
    queue.Enqueue(MakeInferPayload());
  }
  ASSERT_EQ(queue.SizeApprox(), static_cast<size_t>(kBurst));

  std::shared_ptr<Payload> payload;
  std::vector<std::shared_ptr<Payload> > merged_payloads;
  queue.Dequeue(&payload, &merged_payloads);

  ASSERT_EQ(merged_payloads.size(), static_cast<size_t>(kBurst - 1));
  EXPECT_EQ(queue.SizeApprox(), 0u);
  EXPECT_EQ(queue.SizeApprox(), queue.Size());
}

// Runs the untimed WaitForConsumer() on its own thread. The destructor closes
// the queue before joining, so a waiter that was never released fails the
// test instead of hanging it.
class ConsumerWaiter {
 public:
  explicit ConsumerWaiter(InstanceQueue* queue) : queue_(queue)
  {
    std::promise<void> started;
    auto started_future = started.get_future();
    thread_ = std::thread([this, &started]() {
      started.set_value();
      queue_->WaitForConsumer();
      released_.set_value();
    });
    started_future.wait();
    // The thread may not be parked yet; registering late only weakens a test
    // into the immediate-return case, it cannot make one fail.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  ~ConsumerWaiter()
  {
    queue_->Close();
    thread_.join();
  }

  bool Released(std::chrono::milliseconds timeout)
  {
    return released_future_.wait_for(timeout) == std::future_status::ready;
  }

 private:
  InstanceQueue* queue_;
  std::promise<void> released_;
  std::future<void> released_future_{released_.get_future()};
  std::thread thread_;
};

// The consumer count is a lock-free atomic: increments and decrements must
// balance exactly, including through negative values (EnqueuePayload
// decrements before any consumer parks).
TEST(InstanceQueueTest, ConsumerCountArithmetic)
{
  InstanceQueue queue(1 /* max_batch_size */, 0 /* max_queue_delay_ns */);

  EXPECT_EQ(queue.WaitingConsumerCount(), 0);
  queue.DecrementConsumerCount();
  EXPECT_EQ(queue.WaitingConsumerCount(), -1);
  queue.IncrementConsumerCount();
  queue.IncrementConsumerCount();
  EXPECT_EQ(queue.WaitingConsumerCount(), 1);
}

// WaitForConsumer() must return at once when a consumer is already counted;
// no notification can be required.
TEST(InstanceQueueTest, WaitForConsumerReturnsWhenConsumerPresent)
{
  InstanceQueue queue(1 /* max_batch_size */, 0 /* max_queue_delay_ns */);
  queue.IncrementConsumerCount();

  ConsumerWaiter waiter(&queue);
  EXPECT_TRUE(waiter.Released(std::chrono::seconds(5)))
      << "WaitForConsumer() blocked although a consumer was already counted";
}

// Missed-wakeup regression for the waiter registration: a blocked
// WaitForConsumer() must be released by IncrementConsumerCount() from another
// thread, although increments only notify while a waiter is registered.
TEST(InstanceQueueTest, WaitForConsumerWokenByIncrement)
{
  InstanceQueue queue(1 /* max_batch_size */, 0 /* max_queue_delay_ns */);

  ConsumerWaiter waiter(&queue);
  queue.IncrementConsumerCount();
  EXPECT_TRUE(waiter.Released(std::chrono::seconds(5)))
      << "IncrementConsumerCount() failed to wake a blocked WaitForConsumer()";
}

// Several callers can be blocked at once. Each increment wakes one of them, and
// the one still blocked after the first returned must be woken by the next
// increment: waiter registration must be a counter, not a flag cleared by the
// first caller to return.
TEST(InstanceQueueTest, EveryBlockedWaiterIsEventuallyWoken)
{
  InstanceQueue queue(1 /* max_batch_size */, 0 /* max_queue_delay_ns */);

  ConsumerWaiter first(&queue);
  ConsumerWaiter second(&queue);

  queue.IncrementConsumerCount();
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!first.Released(std::chrono::milliseconds(10)) &&
         !second.Released(std::chrono::milliseconds(10)) &&
         std::chrono::steady_clock::now() < deadline) {
  }

  queue.IncrementConsumerCount();
  EXPECT_TRUE(first.Released(std::chrono::seconds(5)));
  EXPECT_TRUE(second.Released(std::chrono::seconds(5)));
}

}  // namespace

using namespace std::chrono_literals;

// Error responses never allocate output buffers, but a response needs an
// allocator to be constructed.
TRITONSERVER_Error*
UnusedResponseAlloc(
    TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id, void* userp, void** buffer, void** buffer_userp,
    TRITONSERVER_MemoryType* actual_memory_type, int64_t* actual_memory_type_id)
{
  return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, "unused");
}

TRITONSERVER_Error*
UnusedResponseRelease(
    TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
    size_t byte_size, TRITONSERVER_MemoryType memory_type,
    int64_t memory_type_id)
{
  return nullptr;
}

void
CountErrorResponse(
    TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp)
{
  TRITONSERVER_Error* error = TRITONSERVER_InferenceResponseError(response);
  if (error != nullptr) {
    static_cast<std::atomic<int>*>(userp)->fetch_add(1);
    TRITONSERVER_ErrorDelete(error);
  }
  TRITONSERVER_InferenceResponseDelete(response);
}

void
DeleteReleasedRequest(
    TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
  TRITONSERVER_InferenceRequestDelete(request);
}

// Public factories create passive instances so tests can control availability
// through RateLimiter::DequeuePayload without a backend thread.
class PayloadSlotWaitTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override
  {
    const auto models =
        std::filesystem::canonical("/proc/self/exe").parent_path() /
        "instance_queue_test_models";
    server_.SetModelRepositoryPaths({(models / "empty").string()});
    server_.SetModelControlMode(ModelControlMode::MODE_EXPLICIT);
    server_.SetPinnedMemoryPoolByteSize(0);
    server_.SetRepoAgentDir((models / "empty").string());
    server_.SetCacheDir((models / "empty").string());
    server_.SetRateLimiterMode(RateLimitMode::RL_OFF);
    server_.SetEnablePeerAccess(false);
    server_.SetResponseCacheEnabled(false);
    const triton::common::BackendCmdlineConfigMap backend_config{
        {"",
         {{"backend-directory", models.string()},
          {"auto-complete-config", "false"},
          {"min-compute-capability", "0"}}}};
    server_.SetBackendCmdlineConfig(backend_config);
    auto status = server_.Init();
    ASSERT_TRUE(status.IsOk()) << status.Message();
    inference::ModelConfig config;
    config.set_name("test");
    config.set_backend("test");
    config.set_max_batch_size(1);
    config.mutable_version_policy()->mutable_latest()->set_num_versions(1);
    auto* group = config.add_instance_group();
    group->set_kind(inference::ModelInstanceGroup::KIND_CPU);
    group->set_count(1);
    group->set_passive(true);
    batcher_config_.add_preferred_batch_size(1);
    batcher_config_.mutable_default_queue_policy()->set_max_queue_size(
        GetParam() ? 0 : 1);
    status = TritonModel::Create(
        &server_, (models / "test").string(), backend_config, {},
        ModelIdentifier("", "test"), 1, config, true, &model_);
    ASSERT_TRUE(status.IsOk()) << status.Message();
    status = TritonModelInstance::CreateInstance(
        model_.get(), "test", TritonModelInstance::Signature(*group, 0),
        TRITONSERVER_INSTANCEGROUPKIND_CPU, 0, {}, true, "", {}, {},
        &instance_);
    ASSERT_TRUE(status.IsOk()) << status.Message();
    ASSERT_TRUE(server_.GetRateLimiter()
                    ->RegisterModelInstance(instance_.get(), {})
                    .IsOk());
  }

  void TearDown() override
  {
    scheduler_.reset();
    instance_.reset();
    model_.reset();
  }

  void FillPrefetchQueue()
  {
    for (int i = 0; i < 2; ++i) {
      ASSERT_TRUE(server_.GetRateLimiter()
                      ->EnqueuePayload(
                          model_.get(), server_.GetRateLimiter()->GetPayload(
                                            Payload::Operation::INFER_RUN))
                      .IsOk());
    }
  }

  std::future<std::shared_ptr<Payload>> Dequeue()
  {
    return std::async(std::launch::async, [this]() {
      std::deque<TritonModelInstance*> instances{instance_.get()};
      std::shared_ptr<Payload> payload;
      server_.GetRateLimiter()->DequeuePayload(instances, &payload);
      return payload;
    });
  }

  void EnqueueRequest()
  {
    auto request = std::make_unique<InferenceRequest>(model_.get(), 1);
    ASSERT_TRUE(request->SetResponseCallback(nullptr, nullptr, nullptr, nullptr)
                    .IsOk());
    ASSERT_TRUE(request->PrepareForInference().IsOk());
    ASSERT_TRUE(scheduler_->Enqueue(request).IsOk());
  }

  // Like EnqueueRequest(), for requests that the scheduler may finish with an
  // error: counts the error responses in 'errors'.
  void EnqueueRejectableRequest(std::atomic<int>* errors)
  {
    static TRITONSERVER_ResponseAllocator* allocator = []() {
      TRITONSERVER_ResponseAllocator* a = nullptr;
      EXPECT_EQ(
          TRITONSERVER_ResponseAllocatorNew(
              &a, UnusedResponseAlloc, UnusedResponseRelease, nullptr),
          nullptr);
      return a;
    }();
    auto request = std::make_unique<InferenceRequest>(model_.get(), 1);
    ASSERT_TRUE(request
                    ->SetResponseCallback(
                        reinterpret_cast<ResponseAllocator*>(allocator),
                        nullptr, CountErrorResponse, errors)
                    .IsOk());
    ASSERT_TRUE(
        request->SetReleaseCallback(DeleteReleasedRequest, nullptr).IsOk());
    ASSERT_TRUE(request->PrepareForInference().IsOk());
    ASSERT_TRUE(scheduler_->Enqueue(request).IsOk());
  }

  void StartBatcher()
  {
    auto status = DynamicBatchScheduler::Create(
        model_.get(), nullptr, 0, true, 1, {}, batcher_config_, &scheduler_);
    ASSERT_TRUE(status.IsOk()) << status.Message();
    // Let the batcher enter its empty-queue wait before the first admission.
    std::this_thread::sleep_for(50ms);
  }

  void StartSlotWait()
  {
    StartBatcher();
    EnqueueRequest();
    // A full prefetch queue suppresses the admission notification. Allow the
    // initial 500 ms scheduler wait to finish before checking the slot wait.
    std::this_thread::sleep_for(600ms);
    EXPECT_EQ(scheduler_->InflightInferenceCount(), 1);
  }

  bool WaitForDispatch()
  {
    const auto deadline = std::chrono::steady_clock::now() + 250ms;
    while (scheduler_->InflightInferenceCount() != 0) {
      if (std::chrono::steady_clock::now() >= deadline) {
        return false;
      }
      std::this_thread::sleep_for(1ms);
    }
    return true;
  }

  InferenceServer server_;
  std::unique_ptr<TritonModel> model_;
  std::shared_ptr<TritonModelInstance> instance_;
  inference::ModelDynamicBatching batcher_config_;
  std::unique_ptr<Scheduler> scheduler_;
};

TEST_P(PayloadSlotWaitTest, BackendAvailabilityWakesSlotWait)
{
  if (GetParam()) {
    FillPrefetchQueue();
  }
  StartSlotWait();
  auto consumer = Dequeue();
  // No later admission releases the blocked batcher.
  EXPECT_TRUE(WaitForDispatch());
  EXPECT_EQ(consumer.wait_for(250ms), std::future_status::ready);
  EXPECT_NE(consumer.get(), nullptr);
}

TEST_P(PayloadSlotWaitTest, ShutdownInterruptsSlotWait)
{
  if (GetParam()) {
    FillPrefetchQueue();
  }
  StartSlotWait();
  auto start = std::chrono::steady_clock::now();
  scheduler_.reset();
  EXPECT_LT(std::chrono::steady_clock::now() - start, 250ms);
}

TEST_P(PayloadSlotWaitTest, StopBeforeSlotWaitDoesNotBlock)
{
  if (GetParam()) {
    FillPrefetchQueue();
  }
  std::stop_source stop;
  stop.request_stop();
  auto start = std::chrono::steady_clock::now();
  EXPECT_FALSE(server_.GetRateLimiter()->PayloadSlotAvailable(
      model_.get(), nullptr, GetParam(), false, 5000000, stop.get_token()));
  EXPECT_LT(std::chrono::steady_clock::now() - start, 250ms);
}

TEST_P(PayloadSlotWaitTest, FirstAdmissionWhileBusyWakesBatcher)
{
  StartBatcher();
  EnqueueRequest();
  std::this_thread::sleep_for(50ms);

  auto consumer = Dequeue();
  EXPECT_EQ(consumer.wait_for(250ms), std::future_status::ready);
  auto payload = consumer.get();
  ASSERT_NE(payload, nullptr);
  EXPECT_EQ(payload->RequestCount(), 1);
}

// The reject pass of the slot wait must run when the closest queued deadline
// expires, not only when the 500 ms window does: without the clamp the 50 ms
// timeout is enforced about 500 ms after the batcher starts waiting.
TEST_P(PayloadSlotWaitTest, RejectTimeoutIsEnforcedDuringSlotWait)
{
  auto* policy = batcher_config_.mutable_default_queue_policy();
  policy->set_timeout_action(inference::ModelQueuePolicy::REJECT);
  policy->set_default_timeout_microseconds(50 * 1000);
  if (GetParam()) {
    FillPrefetchQueue();
  }
  std::atomic<int> errors{0};
  StartBatcher();
  EnqueueRejectableRequest(&errors);
  std::this_thread::sleep_for(250ms);
  EXPECT_EQ(errors.load(), 1);
  EXPECT_EQ(scheduler_->InflightInferenceCount(), 0);
}

TEST_P(PayloadSlotWaitTest, ShutdownInterruptsEmptyQueueWait)
{
  StartBatcher();
  auto start = std::chrono::steady_clock::now();
  scheduler_.reset();
  EXPECT_LT(std::chrono::steady_clock::now() - start, 250ms);
}

TEST_P(PayloadSlotWaitTest, StopRacingWithSlotWaitDoesNotBlock)
{
  if (GetParam()) {
    FillPrefetchQueue();
  }
  for (int i = 0; i < 100; ++i) {
    std::stop_source stop;
    std::promise<void> start;
    auto waiter = std::async(std::launch::async, [&]() {
      start.set_value();
      return server_.GetRateLimiter()->PayloadSlotAvailable(
          model_.get(), nullptr, GetParam(), false, 5000000, stop.get_token());
    });
    start.get_future().wait();
    stop.request_stop();
    EXPECT_EQ(waiter.wait_for(250ms), std::future_status::ready) << i;
    EXPECT_FALSE(waiter.get());
  }
}

INSTANTIATE_TEST_SUITE_P(
    QueuePolicy, PayloadSlotWaitTest, ::testing::Bool(),
    [](const ::testing::TestParamInfo<bool>& info) {
      return info.param ? "Prefetch" : "Finite";
    });

}}  // namespace triton::core
#endif
