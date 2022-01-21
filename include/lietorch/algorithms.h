#ifndef ALGORITHMS_H
#define ALGORITHMS_H

#include "liegroup.h"

namespace lietorch {

template<typename LieGroup>
LieGroup interpolate (const LieGroup &a,
				  const LieGroup &b,
				  float t)
{
	assert (t >= 0 && t <= 1 && "t must be in [0, 1]");

	// corresponds to:
	// a * exp(log (a^-1 * b) * t)
	// (b - a) is an element of the tangent space
	return a + (b - a) * t;
}

template<typename LieGroup, typename clock>
LieGroup extrapolate (const LieGroup &a,
				  const LieGroup &b,
				  const std::chrono::time_point<clock> &timeA,
				  const std::chrono::time_point<clock> &timeB,
				  const std::chrono::time_point<clock> &timeEval)
{
	using DurationSeconds = std::chrono::duration<float, std::ratio<1>>;

	return a + (a - b) * (DurationSeconds (timeEval - timeA).count() /
			DurationSeconds (timeB - timeA).count ());
}

}
#endif // ALGORITHMS_H
