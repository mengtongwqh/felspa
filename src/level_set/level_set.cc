#include <felspa/level_set/level_set.h>

FELSPA_NAMESPACE_OPEN

/* -------- Explicit Instantiations ---------- */
template class ls::LevelSetSurface<dg::AdvectSimulator<1>,
                                   dg::ReinitSimulator<1>>;
template class ls::LevelSetSurface<dg::AdvectSimulator<2>,
                                   dg::ReinitSimulator<2>>;
template class ls::LevelSetSurface<dg::AdvectSimulator<3>,
                                   dg::ReinitSimulator<3>>;
/* ------------------------------------------- */

FELSPA_NAMESPACE_CLOSE
