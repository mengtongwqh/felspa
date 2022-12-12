#ifndef _FELSPA_BASE_UTILITIES_IMPLEMENT_H_
#define _FELSPA_BASE_UTILITIES_IMPLEMENT_H_

#include <felspa/base/utilities.h>

FELSPA_NAMESPACE_OPEN


/* ---------- */
namespace util
/* ---------- */
{
  /* ************************************************** */
  /**
   * This is a struct to test if the types has \c DoFHandler.
   * Can be used to test if a test is a simulator.
   */
  /* ************************************************** */
  namespace internal
  {
    template <typename T>
    struct HasDoFHandler
    {
      constexpr static int dim = T::dimension;

      template <typename U, const dealii::DoFHandler<dim>& (U::*)() const>
      struct SFINAE
      {};

      using YesType = char;
      using NoType = int;

      /** If the SFINAE check fails, return \c NoType. */
      template <typename U>
      static NoType check(...);

      /** If the SFINAE check is successful, return \c YesType. */
      template <typename U>
      static YesType check(SFINAE<U, U::get_dof_handler>*);

      /**
       * \c true if SFINAE check successful, \c false otherwise.
       */
      constexpr static bool value = sizeof(check<T>(0)) == sizeof(YesType);
    };
  }  // namespace internal


  template <typename SimulatorType>
  constexpr bool has_dof_handler(const SimulatorType&)
  {
    return internal::HasDoFHandler<SimulatorType>::value;
  }


  template <typename NumberType>
  std::array<NumberType, 3> hsv_to_rgb(NumberType hue,
                                       NumberType saturation,
                                       NumberType value)
  {
    const NumberType upper_threshold = 0.999;
    const NumberType lower_threshold = 0.001;

    auto clamp = [=](NumberType& x) {
      while (x > 1.0) x -= 1.0;
      while (x < 0.0) x += 1.0;
      if (x > upper_threshold) x = upper_threshold;
      if (x < lower_threshold) x = lower_threshold;
    };

    clamp(hue);
    clamp(saturation);
    clamp(value);

    NumberType h6 = hue * 6.0;
    if (h6 == 6.0) h6 = 0.0;
    int ihue = static_cast<int>(h6);

    NumberType p = value * (1.0 - saturation);
    NumberType q = value * (1.0 - saturation * (h6 - ihue));
    NumberType t = value * (1.0 - saturation * (1.0 - h6 + ihue));

    switch (ihue) {
      case 0:
        return std::array<NumberType, 3>{value, t, p};
      case 1:
        return std::array<NumberType, 3>{q, value, p};
      case 2:
        return std::array<NumberType, 3>{p, value, t};
      case 3:
        return std::array<NumberType, 3>{p, q, value};
      case 4:
        return std::array<NumberType, 3>{t, p, value};
      default:
        return std::array<NumberType, 3>{value, p, q};
    }
  }


  template <typename T>
  std::string demangle_cvr(const char* typeid_name)
  {
    // https://stackoverflow.com/a/20170989
    using TR = typename std::remove_reference<T>::type;
    std::string r = demangle_typeid(typeid_name);

    if (std::is_const<TR>::value) r += " const";
    if (std::is_volatile<TR>::value) r += " volatile";
    if (std::is_lvalue_reference<T>::value)
      r += "&";
    else if (std::is_rvalue_reference<T>::value)
      r += "&&";
    return r;
  }
}  // namespace util

FELSPA_NAMESPACE_CLOSE

#endif  // _FELSPA_BASE_UTILITIES_IMPLEMENT_H_