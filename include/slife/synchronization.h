#ifndef SYNCHRONIZATION_H
#define SYNCHRONIZATION_H

#include <chrono>
#include <deque>
#include <sparcsnode/utils.h>
#include <type_traits>
#include <vector>
#include "definitions.h"
#include "lietorch/algorithms.h"
#include <type_traits>

template<typename T, template<typename ...> class container = std::vector, typename precision = std::chrono::microseconds>
class Signal
{
public:
	using Duration = precision;
	using Clock = std::chrono::system_clock;
	using Time = std::chrono::time_point<Clock, Duration>;
	using Sample = TimedObject<T, Clock, Duration>;
	using DataType = container<Sample>;
	using iterator = typename DataType::iterator;
	using Neighbors = std::pair<boost::optional<Sample>, boost::optional<Sample>>;

	struct Params {
		uint delay;
	};

private:
	Neighbors neighbors (const Time &t) const {
		auto closest = std::lower_bound (timedData.begin(), timedData.end(), t);

		if (closest == timedData.begin ())
			return {boost::none, *closest};
		if (closest == timedData.end ())
			return {*std::prev (closest), boost::none};

		return {*std::prev(closest), *closest};
	}

	T extrapolation (const Sample &first, const Sample &second, const Time &t) const {
		using namespace std::chrono;
		return first.obj() + (second.obj() - first.obj()) * (duration_cast<duration<float>> (t - first.time()).count() /
													 duration_cast<duration<float>>(second.time() - first.time()).count ());
	}

public:
	Signal (const Params &_params = {.delay = 1}):
		 params(_params)
	{}

	void addBack (const Sample &x) {
		timedData.push_back (x);
	}

	template<typename T1 = T, typename = std::enable_if_t<std::is_same<DataType,  std::deque<TimedObject<T1, std::chrono::system_clock, Duration>>>::value>>
	void removeFront () {
		timedData.pop_front ();
	}

	Sample operator[] (int i) const {
		return i >= 0 ? timedData[i] :
				 *prev(timedData.end(), -i);
	}

	Sample &operator[] (int i) {
		return i >= 0 ? timedData[i] :
				 *prev(timedData.end(), -i);
	}

	T at (const Time &t) const
	{
		if (inBounds (t))
			return interpolate (t);
		else {
			COUT("EXTRAPOLATING");
			return extrapolate (t, params.delay);	
		}
	}

	T operator() (const Time &t) const {
		return at (t);
	}

	T operator() (const Duration &afterBegin) {
		return at(timedData.front().time () + afterBegin);
	}

	uint indexAt (const Time &t) const {
		return std::distance (timedData.begin(), prev(std::lower_bound(timedData.begin(), timedData.end(), t)));
	}

	T interpolate (const Time &t) const
	{
		assert (inBounds (t) && "Supplied time out of signal bounds");

		boost::optional<Sample> before, after;
		std::tie (before, after) = neighbors (t);

		return extrapolation (*before, *after, t);
	}

	T extrapolate (const Time &t, uint delay) const
	{
		assert (!inBounds (t) && "Supplied time in signal bounds. Use interpolate instead");
		Sample first, second;

		if (!inLowerBound (t)) {
			first = timedData.front ();
			second = *std::next (timedData.begin (), delay);
		} else {
			first = *std::prev (timedData.end (), delay + 1);
			second = *std::prev (timedData.end ());
		}

		return extrapolation (first, second, t);
	}

	Sample before (const Time &t) const {
		assert (inLowerBound (t) && "Supplied time before signal begin");

		return *neighbors (t).first;
	}

	Sample after(const Time &t) const {
		assert (inUpperBound (t) && "Supplied time after signal end");

		return *neighbors (t).second;
	}

	bool inBounds (const Time &t) const {
		return inLowerBound (t) && inUpperBound (t);
	}

	bool inLowerBound (const Time &t) const {
		return t > timedData.front ().time ();
	}

	bool inUpperBound (const Time &t) const {
		return t < timedData.back().time ();
	}

	size_t size () const {
		return timedData.size ();
	}

	Time timeStart () const {
		return timedData.front().time ();
	}

	Time timeEnd () const {
		return timedData.back().time ();
	}

	typename DataType::const_iterator begin () const {
		return timedData.begin ();
	}

	typename DataType::const_iterator end () const {
		return timedData.end ();
	}

private:
	DataType timedData;
	Params params;
};

template<typename T, template<typename ...> class container>
std::ostream &operator << (std::ostream &os, const Signal<T, container> &sig) {
	for (const typename Signal<T, container>::Sample &curr : sig) {
		os << curr << "\n";
	}

	os << "[ Signal " << TYPE(T) << " {" << sig.size () << "} ]\n";
	return os;
}

class GroundTruthSync
{
public:
	struct Params {
		int queueLength;
		float msOffset;

		DEF_SHARED(Params)
	};

private:
	using GroundTruthSignal = Signal<TargetGroup, std::deque>;
	using Time = typename GroundTruthSignal::Time;
	using Duration = typename GroundTruthSignal::Duration;
	using GroundTruth = typename GroundTruthSignal::Sample;
	using MarkerQueue = std::queue<Time>;

	Params params;
	Duration offset;
	GroundTruthSignal groundTruthSignal;
	MarkerQueue markerQueue;

public:
	GroundTruthSync (const Params &_params);

	bool groundTruthReady ();
	bool markersReady ();
	void updateGroundTruth (const GroundTruth &newGroundTruth);
	void addSynchronizationMarker (const Time &markerTime, boost::optional<float> &expiredMs, boost::optional<float> &futureMs);
	TargetGroup getLastRelativeGroundTruth () const;

	void reset ();
	const MarkerQueue &markers () const {
		return markerQueue;
	}

	DEF_SHARED(GroundTruthSync)
};

template<class Reading>
class ReadingWindow
{
public:
	enum Mode {
		MODE_SLIDING,
		MODE_DOWNSAMPLE
	};

	struct Params {
		Mode mode;
		uint size;

		DEF_SHARED(Params)
	};

private:
	std::queue<Reading> readingQueue;
	uint skipped;
	Params params;

	void addDownsample (const Reading &newReading);
	void addSliding (const Reading &newReading);

public:
	ReadingWindow (const Params &_params);

	void add (const Reading &newReading);
	Reading get ();
	bool isReady () const;
	void reset();

	DEF_SHARED(ReadingWindow)
};

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


class FrequencyEstimator
{
public:
	using FloatSeconds = std::chrono::duration<float>;
	using Clock = std::chrono::system_clock;
	using Time = std::chrono::time_point<Clock, FloatSeconds>;

private:
	Time last;
	FloatSeconds lastPeriod;
	FloatSeconds averagePeriod;
	uint seq;

public:
	FrequencyEstimator ();

	void tick ();
	template<typename OtherTime>
	void tick (const OtherTime &now);

	double estimateSeconds () const;
	double estimateHz () const;
	double lastPeriodSeconds () const;
	void reset();
};

template<typename OtherTime>
void FrequencyEstimator::tick (const OtherTime &now)
{
	Time nowCast = std::chrono::time_point_cast<FloatSeconds> (now);
	if (seq == 0) {
		last = nowCast;
	} else {
		FloatSeconds currentPeriod = nowCast - last;
		lastPeriod = currentPeriod;
		last = nowCast;

		averagePeriod = averagePeriod + 1. / double (seq + 1) * (currentPeriod - averagePeriod);
	}

	seq++;
}

class OffsetEstimator
{
public:
	struct Params {
		bool enable;
		float windowSizeMs;
		uint count;
	};

	using GroundTruthSignal = Signal<TargetGroup>;
	using GroundTruthSample = typename GroundTruthSignal::Sample;
	using Duration = typename GroundTruthSignal::Duration;
	using Time = typename GroundTruthSignal::Time;
	template<typename T>
	using Timed = TimedObject<T, typename GroundTruthSignal::Clock, Duration>;
	using EstimatePeriod = std::pair<TargetGroup, Duration>; // Each estimate coming along with the time period between measurements
	using EstimateSignal = Signal<EstimatePeriod>;
	using EstimateSample = typename EstimateSignal::Sample;

private:
	Params params;
	ReadyFlagsStr flags;
	Duration step;

	EstimateSignal estimateSignal;
	GroundTruthSignal groundTruthSignal;

	torch::Tensor getDelayedGroundTruths(const EstimateSample &estimateSample);
	Duration doEstimation(const torch::Tensor &groundTruthDelays, const torch::Tensor &estimateTensor);

public:
	OffsetEstimator (const Params &_params);

	int totalCount () {
		return 2 * params.count;
	}
	void storeNewEstimate (const TargetGroup &frameEstimate, const Time &firstPointcloudTime, const Time &secondPointcloudTime);
	void storeNewGroundTruth (const GroundTruthSample &groundTruth);
	Duration estimateBestOffset ();

	DEF_SHARED(OffsetEstimator)
};


#endif // SYNCHRONIZATION_H
