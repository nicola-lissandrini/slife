#include "slife/synchronization.h"
#include <algorithm>
#include <chrono>
#include <ratio>


using namespace std;
using namespace std::chrono;
using namespace torch;
using namespace lietorch;


/*****************
 * FrequencyEstimator
 * ***************/

FrequencyEstimator::FrequencyEstimator():
	 seq(0)
{}

void FrequencyEstimator::reset () {
	seq = 0;
}

void FrequencyEstimator::tick () {
	tick (Clock::now ());
}



double FrequencyEstimator::estimateHz() const {
	return 1 / estimateSeconds ();
}

double FrequencyEstimator::lastPeriodSeconds() const {
	return lastPeriod.count ();
}

double FrequencyEstimator::estimateSeconds () const {
	return averagePeriod.count ();
}

/*****************
 * GroundTruthSync
 * ***************/

GroundTruthSync::GroundTruthSync (const GroundTruthSync::Params &_params):
	 params(_params)
{
	offset = chrono::milliseconds((long)params.msOffset);
}

bool GroundTruthSync::groundTruthReady() {
	return groundTruthSignal.size () > 0;
}

bool GroundTruthSync::markersReady() {
	return markerQueue.size () == 2;
}

void GroundTruthSync::updateGroundTruth(const GroundTruth &newGroundTruth)
{
	groundTruthSignal.addBack (newGroundTruth);

	if (groundTruthSignal.size () > params.queueLength)
		groundTruthSignal.removeFront ();
}

void GroundTruthSync::addSynchronizationMarker (const Time &markerTime, boost::optional<float> &expiredMs, boost::optional<float> &futureMs)
{
	Time adjusted = markerTime + offset;
	markerQueue.push (adjusted);

	if (adjusted < groundTruthSignal.timeStart ())
		expiredMs = chrono::duration_cast<chrono::duration<float, std::milli>> (groundTruthSignal.timeStart () - adjusted).count();

	if (adjusted > groundTruthSignal.timeEnd ())
		futureMs = chrono::duration_cast<chrono::duration<float, std::milli>> (adjusted - groundTruthSignal.timeEnd ()).count ();

	if (markerQueue.size() > 2)
		markerQueue.pop ();
}

TargetGroup GroundTruthSync::getLastRelativeGroundTruth() const
{
	TargetGroup first = groundTruthSignal(markerQueue.front ());
	TargetGroup second = groundTruthSignal(markerQueue.back ());

	return first.inverse () * second;
}

/*****************
 * OffsetEstimator
 * ***************/

OffsetEstimator::OffsetEstimator(const OffsetEstimator::Params &_params):
	 params(_params)
{
	step = duration_cast<Duration> (duration<float, std::milli> (float (params.windowSizeMs) / float (totalCount())));
}

void OffsetEstimator::storeNewEstimate (const TargetGroup &estimate, const Time &firstPointcloudTime, const Time &secondPointcloudTime)
{
	if (!params.enable)
		return;

	EstimateSample newSample;
	Duration interval = duration_cast<Duration> (secondPointcloudTime - firstPointcloudTime);

	newSample.obj () = make_pair (estimate, interval);
	newSample.time () = firstPointcloudTime;
	estimateSignal.addBack (newSample);
}

void OffsetEstimator::storeNewGroundTruth (const GroundTruthSample &groundTruth)
{
	if (!params.enable)
		return;

	groundTruthSignal.addBack (groundTruth);
}


OffsetEstimator::Duration OffsetEstimator::doEstimation (const torch::Tensor &groundTruthDelays, const torch::Tensor &estimateTensor)
{
	int index = (estimateTensor.unsqueeze (1)
			   - groundTruthDelays)
				 .norm(2,2)
				 .sum(0)
				 .argmin ()
				 .item().toInt ();

	COUTN(duration_cast<milliseconds> (step).count())
	COUTN(milliseconds (-(int)params.count).count())

	return milliseconds (-(int)params.count) + index * step;
}

torch::Tensor OffsetEstimator::getDelayedGroundTruths (const EstimateSample &estimateSample)
{
	Tensor delayed = torch::empty ({totalCount (), TargetGroup::Dim}, kFloat);

	for (int j = -params.count; j < (signed) params.count; j++) {
		Time currentFirstTime = estimateSample.time() + j * step;
		Time currentSecondTime = estimateSample.time() + estimateSample.obj ().second + j * step;

		TargetGroup firstGroundTruth = groundTruthSignal(currentFirstTime);
		TargetGroup secondGroundTruth = groundTruthSignal(currentSecondTime);

		delayed[j] = (firstGroundTruth.inverse() * secondGroundTruth).coeffs;
	}

	return delayed;
}

OffsetEstimator::Duration OffsetEstimator::estimateBestOffset ()
{
	Tensor groundTruthDelays = torch::empty ({static_cast<long>(estimateSignal.size ()), totalCount(), TargetGroup::Dim}, kFloat);
	Tensor estimateTensor = torch::empty ({static_cast<long>(estimateSignal.size ()), TargetGroup::Dim}, kFloat);

	uint i = 0;
	for (const EstimateSample &currentEstimate : estimateSignal) {
		groundTruthDelays[i] = getDelayedGroundTruths (currentEstimate);
		estimateTensor[i] = currentEstimate.obj ().first.coeffs.flatten();
		i++;
	}

	return doEstimation (groundTruthDelays, estimateTensor);
}


