#include "slife/slife_handler.h"

#include <octomap/OcTree.h>

using namespace std;
using namespace torch;

SlifeHandler::SlifeHandler()
{
	flags.addFlag ("initialized", true);
}

void SlifeHandler::updatePointcloud (const Tensor &pointcloud) {
	optimizer->updatePointcloud(pointcloud);
}

void SlifeHandler::test ()
{
	Test::Type testWhat = tester->getType();
	Tensor testValues;

	testValues = optimizer->test (testWhat);

	if (testValues.numel ())
		tester->publishRangeTensor (testWhat, testValues);
}

int SlifeHandler::synchronousActions ()
{
	test ();
	return 0;
}

void SlifeHandler::init (XmlRpc::XmlRpcValue &xmlParams)
{
	Landscape::Params::Ptr landscapeParams = getLandscapeParams(xmlParams["landscape"]);
	Optimizer::Params::Ptr optimizerParams = getOptimizerParams(xmlParams["optimizer"]);

	optimizer = make_shared<Optimizer> (landscapeParams,
								 optimizerParams);

	params.synthPclSize = paramInt (xmlParams, "synth_pcl_size");

	flags.set ("initialized");
}

Optimizer::Params::Ptr SlifeHandler::getOptimizerParams(XmlRpc::XmlRpcValue &xmlParams)
{
	Optimizer::Params::Ptr optimizerParams = make_shared<Optimizer::Params> ();

	return optimizerParams;
}

Landscape::Params::Ptr SlifeHandler::getLandscapeParams (XmlRpc::XmlRpcValue &xmlParams)
{
	Landscape::Params::Ptr landscapeParams = make_shared<Landscape::Params> ();

	landscapeParams->measureRadius = paramDouble (xmlParams, "measure_radius");
	landscapeParams->smoothRadius = paramDouble (xmlParams, "smooth_radius");
	landscapeParams->precision = paramInt (xmlParams,"precision");

	return landscapeParams;
}













