#ifndef SLIFE_NODE_H
#define SLIFE_NODE_H

#include "sparcsnode.h"
#include "slife/slife_handler.h"
#include "test.h"
#define NODE_NAME "slife"

#include <sensor_msgs/PointCloud2.h>

class SlifeNode : public SparcsNode
{
	SlifeHandler slifeHandler;

	void initParams ();
	void initROS ();
	int actions ();

	void pointcloudCallback (const sensor_msgs::PointCloud2 &pointcloud);

public:
	SlifeNode ();

	friend class Test;
};


MAKE_SHARED (SlifeNode)

#endif // SLIFE_NODE_H
