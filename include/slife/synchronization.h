#ifndef SYNCHRONIZATION_H
#define SYNCHRONIZATION_H

#include <chrono>
#include <sparcsnode/utils.h>
#include "definitions.h"

using Clock = std::chrono::system_clock;
using Duration = std::chrono::duration<double>;
using Time = std::chrono::time_point<Clock, Duration>;

template<class T>
using Timed = TimedClock<T, Clock, Duration>;


template<typename LieGroup, template<typename ...> class container = std::vector>
class Signal
{
	using DataType = container<Timed<LieGroup>>;
	DataType timedData;

public:
	Signal ();

	LieGroup operator[] (uint i);

    Timed<LieGroup> operator() (const Time &t);

	typename DataType::iterator begin () {
		return timedData.begin ();
	}

	typename DataType::iterator end () {
        return timedData.end ();
    }
};

template<typename LieGroup,template<typename ...> typename container>
Timed<LieGroup> Signal<LieGroup, container>::operator()(const Time &t)
{

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
	using GroundTruth = Timed<TargetGroup>;
	using GroundTruthBatch = std::deque<GroundTruth>;
	using MarkerMatch = std::pair<GroundTruth, GroundTruth>;

	GroundTruthBatch groundTruths;
	// The time in the markers correspond to the pcl time
	std::queue<Timed<MarkerMatch>> markerMatches;
	Duration offset;

	Params params;
	GroundTruthBatch::iterator findClosest(const Time &otherTime);
	TargetGroup getMatchingGroundTruth(const Timed<MarkerMatch> &marker) const;

public:
	GroundTruthSync (const Params &_params);

	void updateGroundTruth (const Timed<TargetGroup> &groundTruth);
	void addSynchronizationMarker (const Time &otherTime);
	TargetGroup getRelativeGroundTruth () const;
	TargetGroup getMatchBefore() const;
	TargetGroup getMatchAfter() const;
	bool markersReady () const;
	bool groundTruthReady() const;
	void reset ();

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

class FrequencyEstimator
{

	Time last;
	Duration lastPeriod;
	Duration averagePeriod;
	uint seq;

public:
	FrequencyEstimator ();

	void tick ();
	void tick (const Time &now);
	double estimateSeconds () const;
	double estimateHz () const;
	double lastPeriodSeconds () const;
	void reset();
};

class OffsetEstimator
{
public:
	struct Params {
		bool enable;
		float windowSize;
		uint count;
	};

private:
	using TimedEstimateDuration = std::pair<Timed<TargetGroup>, Duration>;
	using EstimateSignal = std::vector<TimedEstimateDuration>;
	using GroundTruthSignal = std::vector<Timed<TargetGroup>>;

	Params params;
	ReadyFlagsStr flags;

	EstimateSignal estimateSignal;
	GroundTruthSignal groundTruthSignal;

public:
	OffsetEstimator (const Params &_params,
				  const GroundTruthSync::Ptr &_groundTruths);

	void storeNewEstimate (const Timed<TargetGroup> &frameEstimate, const Duration &interval);
	void storeNewGroundTruth (const Timed<TargetGroup> &groundTruth);
	void estimate();

	DEF_SHARED(OffsetEstimator)
};


#endif // SYNCHRONIZATION_H
