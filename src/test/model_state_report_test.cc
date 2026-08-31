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
//

#include "model_repository_manager/model_state_report.h"

#include <string>

#include "gtest/gtest.h"

namespace {

using triton::core::kLeakedReferenceWarningPrefix;
using triton::core::kStuckUnloadReasonPrefix;
using triton::core::ModelReadyState;
using triton::core::ReportedModelState;

constexpr int64_t kSecondNs = 1000000000LL;
constexpr int64_t kThresholdNs = 60 * kSecondNs;

bool
HasPrefix(const std::string& s, const char* prefix)
{
  return s.rfind(prefix, 0) == 0;
}

TEST(ReportedModelStateTest, HealthyReadyPassesThrough)
{
  const auto r = ReportedModelState(
      ModelReadyState::READY, "", 0 /* leaked_refs */,
      -1 /* unloading_for_ns */, kThresholdNs, 0 /* live_refs */);
  EXPECT_EQ(r.first, ModelReadyState::READY);
  EXPECT_EQ(r.second, "");
}

TEST(ReportedModelStateTest, ExistingReasonIsPreserved)
{
  const auto r = ReportedModelState(
      ModelReadyState::UNAVAILABLE, "unloaded", 0, -1, kThresholdNs, 0);
  EXPECT_EQ(r.first, ModelReadyState::UNAVAILABLE);
  EXPECT_EQ(r.second, "unloaded");
}

TEST(ReportedModelStateTest, UnloadingBelowThresholdIsNotStuck)
{
  const auto r = ReportedModelState(
      ModelReadyState::UNLOADING, "", 0, 30 * kSecondNs, kThresholdNs, 5);
  EXPECT_EQ(r.first, ModelReadyState::UNLOADING);
  EXPECT_EQ(r.second, "");
}

TEST(ReportedModelStateTest, UnloadingPastThresholdIsStuckWithLiveRefCount)
{
  const auto r = ReportedModelState(
      ModelReadyState::UNLOADING, "", 0, 120 * kSecondNs, kThresholdNs, 3);
  EXPECT_EQ(r.first, ModelReadyState::UNLOADING);
  EXPECT_TRUE(HasPrefix(r.second, kStuckUnloadReasonPrefix)) << r.second;
  EXPECT_NE(r.second.find("120s ago"), std::string::npos) << r.second;
  EXPECT_NE(r.second.find("3 reference(s) still held"), std::string::npos)
      << r.second;
}

TEST(ReportedModelStateTest, DrainedButUnfinalizedIsReported)
{
  const auto r = ReportedModelState(
      ModelReadyState::UNLOADING, "", 0, 120 * kSecondNs, kThresholdNs, 0);
  EXPECT_TRUE(HasPrefix(r.second, kStuckUnloadReasonPrefix)) << r.second;
  EXPECT_NE(r.second.find("references drained"), std::string::npos) << r.second;
}

TEST(ReportedModelStateTest, ZeroThresholdDisablesTimeoutDetection)
{
  const auto r = ReportedModelState(
      ModelReadyState::UNLOADING, "", 0, 1000 * kSecondNs, 0, 7);
  EXPECT_EQ(r.second, "");
}

TEST(ReportedModelStateTest, UnknownUnloadStartIsNotStuck)
{
  const auto r = ReportedModelState(
      ModelReadyState::UNLOADING, "", 0, -1, kThresholdNs, 7);
  EXPECT_EQ(r.second, "");
}

TEST(ReportedModelStateTest, LeakedReferenceWarnsWhileReady)
{
  const auto r =
      ReportedModelState(ModelReadyState::READY, "", 2, -1, kThresholdNs, 0);
  EXPECT_EQ(r.first, ModelReadyState::READY);
  EXPECT_TRUE(HasPrefix(r.second, kLeakedReferenceWarningPrefix)) << r.second;
  EXPECT_NE(r.second.find("2 leaked reference(s)"), std::string::npos)
      << r.second;
}

TEST(ReportedModelStateTest, LeakedReferenceIsStuckImmediatelyWhenUnloading)
{
  // No threshold wait: a reported leak means the unload can never complete.
  const auto r = ReportedModelState(
      ModelReadyState::UNLOADING, "", 1, 1 * kSecondNs, kThresholdNs, 1);
  EXPECT_TRUE(HasPrefix(r.second, kStuckUnloadReasonPrefix)) << r.second;
  EXPECT_NE(r.second.find("1 leaked reference(s)"), std::string::npos)
      << r.second;
}

TEST(ReportedModelStateTest, LeakedReferenceInOtherStatesPassesThrough)
{
  const auto r =
      ReportedModelState(ModelReadyState::LOADING, "", 4, -1, kThresholdNs, 0);
  EXPECT_EQ(r.second, "");
}

TEST(ReportedModelStateTest, ReasonPrefixesAreAStableContract)
{
  // Documented at TRITONSERVER_ServerModelReportLeakedReference; consumers
  // match on these prefixes.
  EXPECT_STREQ(kStuckUnloadReasonPrefix, "stuck: ");
  EXPECT_STREQ(kLeakedReferenceWarningPrefix, "warning: ");
}

}  // namespace

int
main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
