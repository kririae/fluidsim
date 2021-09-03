//
// Created by kr2 on 8/28/21.
//

#include "mesher.hpp"
#include "particle.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetFilter.h>
#include <openvdb/tools/ParticlesToLevelSet.h>
#include <openvdb/tools/VolumeToMesh.h>

size_t ParticleList::size() const
{
  return data.size();
}

ParticleList::ParticleList(const std::vector<SPHParticle> &_data) : data(_data)
{
}

void ParticleList::getPos(size_t n, openvdb::Vec3R &xyz) const
{
  const auto pos = data[n].pos;
  xyz = openvdb::Vec3R(pos.x, pos.y, pos.z);
}

void ParticleList::getPosRad(size_t n,
                             openvdb::Vec3R &xyz,
                             openvdb::Real &_radius) const
{
  getPos(n, xyz);
  _radius = 0.3 * ::radius;
}

void ParticleList::getAtt(size_t n, openvdb::Index32 &att) const
{
  att = openvdb::Index32(n);
}

void particleToMesh(const std::vector<SPHParticle> &data,
                    std::vector<openvdb::Vec3s> &points,
                    std::vector<openvdb::Vec3I> &triangles,
                    std::vector<openvdb::Vec4I> &quads)
{
  openvdb::initialize();
  double voxelSize = radius / 5.0, halfWidth = 2.0;
  openvdb::FloatGrid::Ptr grid = openvdb::createLevelSet<openvdb::FloatGrid>(
      voxelSize, halfWidth);
  openvdb::tools::ParticlesToLevelSet<openvdb::FloatGrid, openvdb::Index32>
      raster(*grid);

  ParticleList list(data);
  raster.setGrainSize(1);
  raster.rasterizeSpheres(list);
  raster.finalize(true);

  openvdb::tools::LevelSetFilter<openvdb::FloatGrid> filter(*grid);
  filter.gaussian();
  // filter.meanCurvature();

  openvdb::tools::volumeToMesh(*grid, points, triangles, quads, 0.0, 0.5);
}
