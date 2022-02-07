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

template<typename LieGroup, typename time>
LieGroup extrapolate (const LieGroup &a,
				  const LieGroup &b,
				  const time &timeA,
				  const time &timeB,
				  const time &timeEval)
{
	using Duration = typename time::duration;

	return a + (a - b) * (Duration (timeEval - timeA).count() /
			Duration (timeB - timeA).count ());
}

}
#endif // ALGORITHMS_H
