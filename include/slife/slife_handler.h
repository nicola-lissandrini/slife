#ifndef SLIFE_HANDLER_H
#define SLIFE_HANDLER_H

#include <torch/torch.h>
#include <ATen/Tensor.h>
#include <ATen/TensorOperators.h>
#include <ATen/Layout.h>

#include "sparcsnode/common.h"
#include "optimizer.h"
#include "landscape.h"
#include "test.h"

using TargetGroup = lietorch::Pose;

template<class T>
using Timed = TimedClock<T, std::chrono::system_clock>;

class GroundTruthSync
{
public:
	struct Params {
		int queueLength;

		DEF_SHARED(Params)
	};

private:
	using GroundTruth = Timed<TargetGroup>;
	using GroundTruthBatch = std::deque<GroundTruth>;
	using Time = std::chrono::time_point<std::chrono::system_clock>;
	using TimeDuration = std::chrono::duration<float>;
	using MarkerMatch = std::pair<GroundTruth, GroundTruth>;

	GroundTruthBatch groundTruths;
	// The time in the markers correspond to the pcl time
	std::queue<Timed<MarkerMatch>> markerMatches;

	Params params;
	GroundTruthBatch::iterator findClosest(const Time &otherTime);
	TargetGroup getMatchingGroundTruth(const Timed<MarkerMatch> &marker) const;

public:
	GroundTruthSync (const Params &_params);

	void updateGroundTruth (const Timed<TargetGroup> &groundTruth);
	void addSynchronizationMarker (const Time &otherTime);
	TargetGroup getRelativeGroundTruth () const;
	bool markersReady () const;
	bool groundTruthReady() const;

	DEF_SHARED(GroundTruthSync)
};

class SlifeHandler
{
public:
	enum OutputTensorType {
		OUTPUT_ESTIMATE,
		OUTPUT_DEBUG_1,
		OUTPUT_DEBUG_2
	};

	enum TargetOptimizationGroup {
		TARGET_POSITION,
		TARGET_QUATERNION_R4,
		TARGET_QUATERNION,
		TARGET_POSE_R4,
		TARGET_POSE,
		TARGET_DUAL_QUATERNION
	};

private:
	struct Params {
		bool syntheticPcl;
		TargetOptimizationGroup targetOptimizationGroup;
		GroundTruthSync::Params groundTruthTracker;

		DEF_SHARED (Params)
	};
	
	using TensorPublisherExtra = std::function<void (OutputTensorType, const torch::Tensor &, const std::vector<uint8_t> &extraData)>;
	using TensorPublisher = std::function<void (OutputTensorType, const torch::Tensor &)>;

	Params params;
	ReadyFlags<std::string> flags;
	typename PointcloudMatchOptimizer<TargetGroup>::Ptr optimizer;
	GroundTruthSync::Ptr groundTruthSync;
	TensorPublisher tensorPublishCallback;
	TensorPublisherExtra tensorPublishExtraCallback;

	Tensor computeHistoryError (const std::vector<TargetGroup> &historyVector, const TargetGroup &groundTruth);
	Tensor historyToTensor (const std::vector<TargetGroup> &historyVector);
	template<class LieGroup>
	LieGroup poseTensorToGroup (const torch::Tensor &poseTensor) const;

	typename PointcloudMatchOptimizer<TargetGroup>::Params::Ptr getOptimizerParams (XmlRpc::XmlRpcValue &xmlParams);
	typename PointcloudMatch<TargetGroup>::Params::Ptr getCostFunctionParams (XmlRpc::XmlRpcValue &xmlParams);

	Landscape::Params::Ptr getLandscapeParams (XmlRpc::XmlRpcValue &xmlParams);
	SlifeHandler::Params getHandlerParams(XmlRpc::XmlRpcValue &xmlParams);

	void test ();

public:
	SlifeHandler(const TensorPublisherExtra &_tensorPublisher);

	void init (XmlRpc::XmlRpcValue &xmlParams);
	void updatePointcloud (const Timed<Tensor> &timedPointcloud);
	void updateGroundTruth (const Timed<Tensor> &timedGroundTruthTensor);

	bool isSyntheticPcl() const;
	bool isReady () const;

	int synchronousActions ();

	DEF_SHARED (SlifeHandler)
};


#endif // SLIFE_HANDLER_H
