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

#include "infer_request.h"

#include <memory>

#include "gtest/gtest.h"
#include "model.h"

namespace triton { namespace core {
namespace {

// Builds a minimal Model that passes Model::Init() validation so that
// InferenceRequest's constructor-time SetPriority(0) call (which reads
// MaxPriorityLevel()/DefaultPriorityLevel() from the model) does not read
// uninitialized memory. No dynamic_batching/sequence_batching/ensemble
// section is configured, so Model::Init() takes the "else" branch and
// deterministically sets both priority levels to 0. No inputs/outputs are
// configured either, so ValidateModelIOConfig() trivially passes and
// AddOriginalInput() (which does not consult the model) is unaffected.
std::shared_ptr<Model>
MakeTestModel()
{
  inference::ModelConfig config;
  config.set_name("request_reuse_test_model");
  // An unrecognized backend name makes ValidateModelConfig() skip the
  // platform/backend consistency check.
  config.set_backend("unknown_test_backend");
  config.mutable_version_policy();

  auto model = std::make_shared<Model>(
      0.0 /* min_compute_capability */, "" /* model_dir */,
      ModelIdentifier("", "request_reuse_test_model"), 1 /* version */,
      config);
  EXPECT_TRUE(model->Init(true /* is_config_provided */).IsOk());
  return model;
}

// Reset() must clear inputs, parameters and identity/config fields back to
// the values a freshly-constructed InferenceRequest would have, and must do
// so completely enough that re-adding an input under a name used before
// Reset() succeeds (i.e. the map was actually cleared, not just shadowed).
TEST(RequestReuseTest, ResetClearsState)
{
  auto model = MakeTestModel();
  InferenceRequest request(model, 1 /* requested_model_version */);

  ASSERT_TRUE(
      request.AddOriginalInput("input0", inference::DataType::TYPE_FP32, {1})
          .IsOk());
  ASSERT_TRUE(
      request.AddOriginalInput("input1", inference::DataType::TYPE_FP32, {1})
          .IsOk());
  request.SetId("request-1");
  request.SetPriority(5);
  request.SetTimeoutMicroseconds(1000);
  request.SetCorrelationId(InferenceRequest::SequenceId(uint64_t(42)));
  ASSERT_TRUE(request.AddParameter("param0", "value0").IsOk());

  // INITIALIZED -> RELEASED is a no-op transition allowed by the state
  // machine, sufficient to exercise Reset()'s state precondition without
  // pulling in the full release callback machinery.
  ASSERT_TRUE(request.SetState(InferenceRequest::State::RELEASED).IsOk());

  ASSERT_TRUE(request.Reset(model, 1).IsOk());

  EXPECT_TRUE(request.OriginalInputs().empty());
  EXPECT_TRUE(request.Parameters().empty());
  EXPECT_EQ(request.Priority(), 0u);
  EXPECT_EQ(request.TimeoutMicroseconds(), 0u);
  EXPECT_EQ(request.CorrelationId().UnsignedIntValue(), 0u);
  EXPECT_TRUE(request.Id().empty());

  EXPECT_TRUE(
      request.AddOriginalInput("input0", inference::DataType::TYPE_FP32, {1})
          .IsOk());
}

// Reset() must rebind the request to the model/version it is given so a
// reload or version switch between reuses is honored (the model pointer
// itself has no public accessor; the version rebind exercises the same
// assignment block).
TEST(RequestReuseTest, ResetRebindsModelVersion)
{
  auto model = MakeTestModel();
  InferenceRequest request(model, 1);

  ASSERT_TRUE(request.SetState(InferenceRequest::State::RELEASED).IsOk());
  ASSERT_TRUE(request.Reset(model, 2).IsOk());

  EXPECT_EQ(request.RequestedModelVersion(), 2);
}

// Reset() must reject requests that are actually in flight.
TEST(RequestReuseTest, ResetWrongStateFails)
{
  auto model = MakeTestModel();
  InferenceRequest request(model, 1);

  ASSERT_TRUE(request.SetState(InferenceRequest::State::PENDING).IsOk());
  ASSERT_TRUE(request.SetState(InferenceRequest::State::EXECUTING).IsOk());

  EXPECT_FALSE(request.Reset(model, 1).IsOk());
}

// When the request is the sole owner of its response factory,
// SetResponseFactory() must reinitialize the existing object in place
// instead of allocating a new one.
TEST(RequestReuseTest, FactoryReuseWhenSoleOwner)
{
  auto model = MakeTestModel();
  InferenceRequest request(model, 1);
  ASSERT_TRUE(
      request.SetResponseCallback(nullptr, nullptr, nullptr, nullptr).IsOk());

  request.SetResponseFactory();
  const auto* first = request.ResponseFactory().get();

  request.SetResponseFactory();
  const auto* second = request.ResponseFactory().get();

  EXPECT_EQ(first, second);
}

// When another owner (e.g. a backend holding the factory via
// TRITONBACKEND_ResponseFactoryNew) still references the current response
// factory, SetResponseFactory() must allocate a new object and leave the
// shared one untouched.
TEST(RequestReuseTest, FactoryFreshWhenShared)
{
  auto model = MakeTestModel();
  InferenceRequest request(model, 1);
  ASSERT_TRUE(
      request.SetResponseCallback(nullptr, nullptr, nullptr, nullptr).IsOk());

  request.SetResponseFactory();
  std::shared_ptr<InferenceResponseFactory> held = request.ResponseFactory();
  const auto* first = held.get();

  request.SetResponseFactory();
  const auto* second = request.ResponseFactory().get();

  EXPECT_NE(first, second);
  // The still-referenced factory must not have been reinitialized as a side
  // effect of creating the new one.
  EXPECT_FALSE(held->IsCancelled());
}

// A cancellation on the response factory must not survive into the factory
// object reused for the next inference.
TEST(RequestReuseTest, CancelledFactoryCleared)
{
  auto model = MakeTestModel();
  InferenceRequest request(model, 1);
  ASSERT_TRUE(
      request.SetResponseCallback(nullptr, nullptr, nullptr, nullptr).IsOk());

  request.SetResponseFactory();
  ASSERT_TRUE(request.Cancel().IsOk());
  EXPECT_TRUE(request.IsCancelled());

  // Sole owner here, so this reinitializes the same factory object in place.
  request.SetResponseFactory();
  EXPECT_FALSE(request.IsCancelled());
}

}  // namespace
}}  // namespace triton::core
