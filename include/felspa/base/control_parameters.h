#ifndef _FELSPA_BASE_CONTROL_PARAMETERS_H_
#define _FELSPA_BASE_CONTROL_PARAMETERS_H_

#include <deal.II/base/parameter_handler.h>

#include <string>

class ControlBase
{
 protected:
  /** Constructor */
  ControlBase(const std::string& id_string) : subsection_id_string(id_string){};

  /** Destructor */
  virtual ~ControlBase() = default;

  virtual void declare_parameters(dealii::ParameterHandler& prm) = 0;

  virtual void parse_parameters(dealii::ParameterHandler& prm) = 0;

  std::string subsection_id_string;
};

#endif  // _FELSPA_BASE_CONTROL_PARAMETERS_H_ //
