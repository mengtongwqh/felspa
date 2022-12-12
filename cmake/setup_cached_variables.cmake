# This file setup some cached global variables for the build system

# ------------------------------------------------
# PROMOTE ALL COMPILER/LINKER FLAGS
# ------------------------------------------------
promote_to_cache(FELSPA_CXX_FLAGS_DEBUG STRING
  "debug compiler flags for felspa library")

promote_to_cache(FELSPA_CXX_FLAGS_RELEASE STRING
  "release compiler flags for felspa library")

promote_to_cache(FESLPA_LINKER_FLAGS STRING
  "Compilation flags for FELSPA Library")
