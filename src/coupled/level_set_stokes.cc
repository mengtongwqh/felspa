
#include <felspa/coupled/level_set_stokes.h>

FELSPA_NAMESPACE_OPEN

/* --------- Explicit Instantiations --------- */
// Note that stokes solver is only implemented for dimension = 2,3
template class LevelSetStokes<dg::AdvectSimulator<2, types::DoubleType>,
                              dg::ReinitSimulator<2, types::DoubleType>>;
template class LevelSetStokes<dg::AdvectSimulator<3, types::DoubleType>,
                              dg::ReinitSimulator<3, types::DoubleType>>;

template class internal::LevelSetStokesParticles<2, types::DoubleType>;
template class internal::LevelSetStokesParticles<3, types::DoubleType>;
/* ------------------------------------------- */

FELSPA_NAMESPACE_CLOSE
