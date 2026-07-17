// Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "staging_queue.h"

#include <atomic>
#include <thread>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"

namespace triton { namespace core {
namespace {

// Instrumented item used in place of InferenceRequest: tags each item with
// its producer id and a per-producer sequence number, and counts
// destructions so ownership can be verified.
struct TestItem {
  TestItem(int producer_id, uint64_t seq)
      : producer_id(producer_id), seq(seq)
  {
  }
  ~TestItem() { destruction_count.fetch_add(1, std::memory_order_relaxed); }

  int producer_id;
  uint64_t seq;

  static std::atomic<int> destruction_count;
};
std::atomic<int> TestItem::destruction_count{0};

TEST(StagingQueueTest, FifoOrderSingleProducer)
{
  StagingQueue<TestItem> queue;
  constexpr int kNumItems = 100;
  for (int i = 0; i < kNumItems; ++i) {
    queue.Push(std::make_unique<TestItem>(0, i));
  }

  auto items = queue.TakeAll();
  ASSERT_EQ(items.size(), static_cast<size_t>(kNumItems));
  for (int i = 0; i < kNumItems; ++i) {
    EXPECT_EQ(items[i]->seq, static_cast<uint64_t>(i));
  }
}

TEST(StagingQueueTest, SizeAndEmptyTrackPushAndTakeAll)
{
  StagingQueue<TestItem> queue;
  EXPECT_TRUE(queue.Empty());
  EXPECT_EQ(queue.Size(), 0u);

  queue.Push(std::make_unique<TestItem>(0, 0));
  queue.Push(std::make_unique<TestItem>(0, 1));
  EXPECT_FALSE(queue.Empty());
  EXPECT_EQ(queue.Size(), 2u);

  auto items = queue.TakeAll();
  EXPECT_EQ(items.size(), 2u);
  EXPECT_TRUE(queue.Empty());
  EXPECT_EQ(queue.Size(), 0u);

  // TakeAll on an empty queue returns an empty deque and is reentrant.
  auto empty_items = queue.TakeAll();
  EXPECT_TRUE(empty_items.empty());
  auto empty_items2 = queue.TakeAll();
  EXPECT_TRUE(empty_items2.empty());
}

TEST(StagingQueueTest, ItemsDroppedWithQueueDestructionAreDestroyedOnce)
{
  const int before = TestItem::destruction_count.load();
  {
    StagingQueue<TestItem> queue;
    queue.Push(std::make_unique<TestItem>(0, 0));
    queue.Push(std::make_unique<TestItem>(0, 1));
    queue.Push(std::make_unique<TestItem>(0, 2));
    // Queue destroyed here without a TakeAll(); items must be destroyed
    // exactly once as part of the queue's own destruction.
  }
  EXPECT_EQ(TestItem::destruction_count.load() - before, 3);
}

TEST(StagingQueueTest, ConcurrentProducersSingleDrainer)
{
  constexpr int kNumProducers = 8;
  constexpr uint64_t kItemsPerProducer = 50000;
  constexpr uint64_t kTotalItems = kNumProducers * kItemsPerProducer;

  StagingQueue<TestItem> queue;
  std::atomic<bool> stop_producing{false};

  std::vector<std::thread> producers;
  for (int p = 0; p < kNumProducers; ++p) {
    producers.emplace_back([&queue, p]() {
      for (uint64_t seq = 0; seq < kItemsPerProducer; ++seq) {
        queue.Push(std::make_unique<TestItem>(p, seq));
      }
    });
  }

  // Per-producer count of received items and the last sequence number seen,
  // used to verify strictly increasing per-producer sequences and to detect
  // duplicates.
  std::vector<uint64_t> received_count(kNumProducers, 0);
  std::vector<int64_t> last_seq(kNumProducers, -1);
  uint64_t total_received = 0;

  std::thread drainer([&]() {
    while (total_received < kTotalItems) {
      auto items = queue.TakeAll();
      for (auto& item : items) {
        EXPECT_GT(static_cast<int64_t>(item->seq), last_seq[item->producer_id]);
        last_seq[item->producer_id] = item->seq;
        ++received_count[item->producer_id];
        ++total_received;
      }
    }
  });

  for (auto& t : producers) {
    t.join();
  }
  drainer.join();

  EXPECT_EQ(total_received, kTotalItems);
  for (int p = 0; p < kNumProducers; ++p) {
    EXPECT_EQ(received_count[p], kItemsPerProducer);
    EXPECT_EQ(last_seq[p], static_cast<int64_t>(kItemsPerProducer) - 1);
  }
}

}  // namespace
}}  // namespace triton::core
