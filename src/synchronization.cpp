#include "slife/synchronization.h"

using namespace std;
using namespace torch;
using namespace lietorch;

FrequencyEstimator::FrequencyEstimator():
	seq(0)
{}

void FrequencyEstimator::reset () {
	seq = 0;
}

void FrequencyEstimator::tick () {
	tick (Clock::now ());
}

void FrequencyEstimator::tick (const Time &now)
{
	if (seq == 0) {
		last = now;
	} else {
		Duration currentPeriod = now - last;
		lastPeriod = currentPeriod;
		last = now;

		averagePeriod = averagePeriod + 1. / double (seq + 1) * (currentPeriod - averagePeriod);
	}

	seq++;
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

GroundTruthSync::GroundTruthSync (const GroundTruthSync::Params &_params):
	params(_params)
{
	offset = chrono::duration<float, std::milli> (params.msOffset);
}

void GroundTruthSync::updateGroundTruth (const Timed<TargetGroup> &groundTruth)
{
	groundTruths.push_back (groundTruth);

	if (groundTruths.size () > params.queueLength)
		groundTruths.pop_front ();
}


void GroundTruthSync::addSynchronizationMarker (const Time &otherTime)
{
	GroundTruthBatch::iterator closest;
	Timed<MarkerMatch> newMarker;
	Time otherTimeAdjusted = otherTime + offset;

	newMarker.time () = otherTimeAdjusted;

	if (otherTimeAdjusted < groundTruths.front ().time ()) {
		ROS_WARN_STREAM ("Ground truth matching the supplied timestamp has expired by " <<
					  (chrono::duration<float, std::milli> (groundTruths.front ().time () - otherTimeAdjusted)).count() << "ms.\n"
																								"Using last ground truth stored, probabily outdated.\n"
																								"Consider increasing 'ground_truth_queue_length' to avoid this issue");
		closest = groundTruths.begin ();
	} else
		closest = findClosest (otherTimeAdjusted);

	if (next (closest) == groundTruths.end ())
		newMarker.obj () = make_pair (*prev (closest), *closest);
	else
		newMarker.obj () = make_pair (*closest, *next (closest));

	markerMatches.push (newMarker);

	if (markerMatches.size () > 2)
		markerMatches.pop ();
}

GroundTruthSync::GroundTruthBatch::iterator GroundTruthSync::findClosest (const Time &otherTime)
{
	auto it = std::lower_bound (groundTruths.begin (), groundTruths.end (), otherTime, [] (GroundTruthBatch::const_reference gt, decltype(otherTime) ot){ return gt.time () < ot; });

	if (it == groundTruths.end ())
		return prev (it);

	return it;
}

TargetGroup GroundTruthSync::getMatchingGroundTruth (const Timed<MarkerMatch> &marker) const
{
	GroundTruth before = marker.obj ().first;
	GroundTruth after = marker.obj ().second;

	return extrapolate (before.obj (), after.obj (),
					before.time (), after.time (), marker.time ());
}

TargetGroup GroundTruthSync::getRelativeGroundTruth () const {
	TargetGroup before = getMatchBefore ();
	TargetGroup after  = getMatchAfter ();

	return before.inverse() * after;
}

TargetGroup GroundTruthSync::getMatchBefore () const {
	return getMatchingGroundTruth (markerMatches.front ());
}

TargetGroup GroundTruthSync::getMatchAfter () const {
	return getMatchingGroundTruth (markerMatches.back ());
}

bool GroundTruthSync::markersReady() const {
	return markerMatches.size () == 2;
}

bool GroundTruthSync::groundTruthReady () const {
	return groundTruths.size () > 1;
}

void GroundTruthSync::reset()
{
	groundTruths.clear ();
	markerMatches = {};
}

template<class Reading>
ReadingWindow<Reading>::ReadingWindow(const ReadingWindow::Params &_params):
	params(_params),
	skipped(_params.size)
{
}

template<class Reading>
void ReadingWindow<Reading>::reset ()
{
	readingQueue = {};
	skipped = params.size;
}

template<class Reading>
void ReadingWindow<Reading>::addDownsample (const Reading &newReading)
{
	if (skipped == params.size) {
		skipped = 0;
		if (readingQueue.size () > 0)
			readingQueue.pop ();
		readingQueue.push (newReading);
	} else
		skipped++;
}

template<class Reading>
void ReadingWindow<Reading>::addSliding (const Reading &newReading)
{
	readingQueue.push (newReading);

	if (readingQueue.size () == params.size)
		readingQueue.pop ();
}

template<class Reading>
void ReadingWindow<Reading>::add(const Reading &newReading)
{
	switch (params.mode) {
	case MODE_DOWNSAMPLE:
		addDownsample (newReading);
		break;
	case MODE_SLIDING:
		addSliding (newReading);
		break;
	}
}

template<class Reading>
Reading ReadingWindow<Reading>::get () {
	return readingQueue.front ();
}

template<class Reading>
bool ReadingWindow<Reading>::isReady() const
{
	switch (params.mode) {
	case MODE_DOWNSAMPLE:
		return skipped == 0;
	case MODE_SLIDING:
		return readingQueue.size () == params.size;
	}
}

OffsetEstimator::OffsetEstimator(const OffsetEstimator::Params &_params, const GroundTruthSync::Ptr &_groundTruths):
	params(_params),
	groundTruths(_groundTruths)
{
	flags.addFlag ("first_estimate");
}

void OffsetEstimator::storeNewEstimate (const Timed<TargetGroup> &estimate, const Duration &interval)
{
	if (!params.enable)
		return;

	estimateSignal.push_back (make_pair (estimate, interval));
}

void OffsetEstimator::storeNewGroundTruth (const Timed<TargetGroup> &groundTruth)
{
	if (!params.enable)
		return;

	groundTruthSignal.push_back (groundTruth);
}

void OffsetEstimator::getDelayedGroundTruths (Tensor &delayed, const Timed<TargetGroup> &timedEstimate, const Duration &duration)
{
	GroundTruthSignal::iterator matchingGroundTruth = std::lower_bound (groundTruthSignal.begin (),
														   groundTruthSignal.end (),
														   [](const auto &gt, const auto &tm) { return gt.time() < tm; });

}

void OffsetEstimator::estimate ()
{
	Tensor groundTruthDelays = torch::empty ({estimateSignal.size (), params.count, TargetGroup::Dim}, kFloat);

	for (TimedEstimateDuration &currentEstimate : estimateSignal) {
		Timed<TargetGroup> timedCurrEstimate = currentEstimate.first;
		Duration currDuration = currentEstimate.second;

		getDelayedGroundTruths (groundTruthDelays, timedCurrEstimate, currDuration);
	}
}




