#include "../include/lietorch/rn.h"
#include <boost/none.hpp>
#include <deque>
#include <ostream>
#include <ratio>
#include <type_traits>
#include <chrono>
#include <vector>
#include "../../sparcsnode/include/sparcsnode/utils.h"
#include "../include/lietorch/algorithms.h"

template<class to_duration = std::chrono::milliseconds, class time_point>
std::string printTime (const time_point &time) {
    using TimeInt = std::chrono::time_point<typename time_point::clock>;
    TimeInt timeInt = std::chrono::time_point_cast<typename TimeInt::duration> (time);
    auto coarse = std::chrono::system_clock::to_time_t(timeInt);
    auto fine = std::chrono::time_point_cast<to_duration>(time);

    char buffer[sizeof "9999-12-31 23:59:59.999"];

    std::snprintf(buffer + std::strftime(buffer, sizeof buffer - 3,
                                         "%F %T.", std::localtime(&coarse)),
                  4, "%03lu", fine.time_since_epoch().count() % 1000);
    return buffer;
}

#define TYPE(type) (abi::__cxa_demangle(typeid(type).name(), NULL,NULL,NULL))


template<class T, class clock, class duration = std::chrono::duration<float>>
class TimedObject
{
public:
    using Time = std::chrono::time_point<clock, duration>;

    template<typename other_clock, typename ...Args>
    TimedObject (const std::chrono::time_point<other_clock> &time, Args &&...args):
        _obj(args ...),
        _time(std::chrono::time_point_cast<duration> (time))
    {}

    TimedObject () = default;

    // Avoid implicit direct conversion from T to Timed<T>
    TimedObject (const T &other) = delete;

    T &obj() {
        return _obj;
    }

    const T &obj () const {
        return _obj;
    }

    Time &time () {
        return _time;
    }

    const Time &time () const {
        return _time;
    }


#define TIMED_TIMED_COMPARISON(op) \
    bool operator op (const TimedObject<T, clock, duration> &rhs) const { \
            return time () op rhs.time (); \
    }

#define TIMED_TIME_COMPARISON(op) \
    bool operator op (const Time &rhs) const { \
            return time () op rhs; \
    }

    TIMED_TIMED_COMPARISON(<)
    TIMED_TIMED_COMPARISON(<=)
    TIMED_TIMED_COMPARISON(>)
    TIMED_TIMED_COMPARISON(>=)
    TIMED_TIME_COMPARISON(<)
    TIMED_TIME_COMPARISON(<=)
    TIMED_TIME_COMPARISON(>)
    TIMED_TIME_COMPARISON(>=)

private:
    T _obj;
    Time _time;

};

template<class T, class clock, class duration>
std::ostream &operator << (std::ostream &os, const TimedObject<T, clock, duration> &timed) {
    os << "[" << printTime (timed.time()) << "] " << timed.obj ();
    return os;
}

using namespace std;
using namespace std::chrono_literals;

template<typename T, template<typename ...> class container = std::vector, typename precision = std::chrono::microseconds>
class Signal
{
public:
    using Time = std::chrono::time_point<std::chrono::system_clock, precision>;
    using Sample = TimedObject<T, std::chrono::system_clock, precision>;
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

    template<typename T1 = T, typename = std::enable_if_t<std::is_same<DataType,  std::deque<TimedObject<T1, std::chrono::system_clock, precision>>>::value>>
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
        else
            return extrapolate (t, params.delay);
    }

    T operator() (const Time &t) const {
        return at (t);
    }

    template<typename duration>
    T operator() (const duration &afterBegin) {
        return at(timedData.front().time () + std::chrono::duration_cast<precision> ((duration) afterBegin));
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

int main () {

    using namespace std::chrono_literals;
    using namespace std::chrono;

    Signal<float> x;
    using Sample = Signal<float>::Sample;

    auto now = system_clock::now ();
    Sample a(now, 1);
    x.addBack (a);
    Sample b(now + 10ms, 2);
    x.addBack (b);
    Sample c(now + 20ms, 4);
    x.addBack (c);

    cout << x(22ms) << endl;
}

