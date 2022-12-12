#ifndef _FELSPA_BASE_UTILITIES_H_
#define _FELSPA_BASE_UTILITIES_H_

#include <cxxabi.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_handler.h>
#include <felspa/base/felspa_config.h>

#include <algorithm>
#include <cmath>
#include <type_traits>

FELSPA_NAMESPACE_OPEN

/* ---------- */
namespace util
/* ---------- */
{
  /* ************************************************** */
  /**
   * Shamelessly import the to_string function family from \c deal.II
   */
  /* ************************************************** */
  using dealii::Utilities::int_to_string;
  using dealii::Utilities::to_string;


  /* ************************************************** */
  /**
   * Return the value of underlying type of
   * a variable of an enumeration type
   */
  /* ************************************************** */
  template <typename Enumeration>
  auto enum_underlying_type(Enumeration obj)
    -> std::underlying_type_t<Enumeration>
  {
    return static_cast<std::underlying_type_t<Enumeration>>(obj);
  }


  /* ************************************************** */
  /**
   * Useful for friend declaration in classes with
   * template-template paramter
   */
  /* ************************************************** */
  template <template <typename> class U>
  struct identity
  {
    template <typename T>
    using type = U<T>;
  };


  /* ************************************************** */
  /**
   * find minimum element in a built-in array
   */
  /* ************************************************** */
  template <typename T, std::size_t N>
  T array_min(T (&arr)[N]) noexcept
  {
    return *(std::min_element(arr, arr + N));
  }


  /* ************************************************** */
  /**
   * find maximum element in a built-in array
   */
  /* ************************************************** */
  template <typename T, std::size_t N>
  T array_max(T (&arr)[N]) noexcept
  {
    return *(std::max_element(arr, arr + N));
  }


  /* ************************************************** */
  /**
   * A simple function to evaluate the norm of an \c initializer_list
   * we do l2-norm by default
   */
  /* ************************************************** */
  template <typename T>
  T norm(const std::initializer_list<T>& a)
  {
    T val = T();
    for (const auto& elmt : a) val += elmt * elmt;
    return sqrt(val);
  }


  /* ************************************************** */
  /**
   * A type-safe signum function.
   */
  /* ************************************************** */
  template <typename T>
  int sign(T val)
  {
    return (val > T()) - (val < T());
  }


  /* ************************************************** */
  /**
   * For a filename with/without path
   * \return the extension of the file.
   */
  /* ************************************************** */
  std::string get_file_extension(const std::string& filename);


  /* ************************************************** */
  /**
   * Test if an object contains \c dealii::DoFHandler
   */
  /* ************************************************** */
  template <typename SimulatorType>
  constexpr bool has_dof_handler(const SimulatorType&);


  /* ************************************************** */
  /**
   * Digest the typename string obtained from typeid(T).name()
   */
  /* ************************************************** */
  std::string demangle_typeid(const char* typeid_name);

#define FELSPA_DEMANGLE(var) \
  FELSPA_NAMESPACE::util::demangle_typeid(typeid(var).name())


  /* ************************************************** */
  /**
   * Tp be used with FELSPA_DEMANGLE macro
   * to allow both RTTI (runtime type info)
   * and const/volatile/reference information.
   */
  /* ************************************************** */
  template <typename T>
  std::string demangle_cvr(const char* typeid_name);

#define FELSPA_DEMANGLE_CVR(var) \
  FELSPA_NAMESPACE::util::demangle_cvr<decltype(var)>(typeid(var).name())


  /* ************************************************** */
  /**
   * Given the HSV values, convert it into an array of RGB
   * @return std::array<unsigned int, 3>
   */
  /* ************************************************** */
  template <typename NumberType>
  std::array<NumberType, 3> hsv_to_rgb(NumberType hue,
                                       NumberType saturation,
                                       NumberType value);

  /* ************************************************** */
  /**
   * @brief Get the date and time in a string
   * @param format_string
   */
  /* ************************************************** */
  std::string get_date_time(const std::string& format_string = "%Y%m%d_%H%M%S");

}  // namespace util

FELSPA_NAMESPACE_CLOSE

#include "src/utilities.implement.h"

#endif /* _FELSPA_BASE_UTILITIES_H_ */
