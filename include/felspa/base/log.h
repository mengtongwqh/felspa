#ifndef _FELSPA_BASE_LOG_H_
#define _FELSPA_BASE_LOG_H_

#include <deal.II/base/logstream.h>
#include <felspa/base/felspa_config.h>

FELSPA_NAMESPACE_OPEN

/**
 * \class FelspaLog
 * \brief A bloody hack of the dealii::LogStream class
 * to implement a global logging mechanism for FELSPA library.
 */
class FelspaLog : public dealii::LogStream
{
 public:
  FelspaLog() : dealii::LogStream()
  {
    pop();
    push("FELSPA");
  }
};

/* --------------------------------------------------*/
/**
 * Declaration for global logger for FELSPA.
 * The global logger is defined in log.cc file
 */
extern FelspaLog felspa_log;
/* --------------------------------------------------*/

FELSPA_NAMESPACE_CLOSE

/* --------------------------------------------------*/
/**
 * \brief macro LOG_PREFIX
 * Use this macro to set prefix to global log object
 */
#define LOG_PREFIX(prfx)                           \
  FELSPA_NAMESPACE::FelspaLog::Prefix prefix(prfx, \
                                             FELSPA_NAMESPACE::felspa_log);

#endif  // _FELSPA_BASE_LOG_H_
