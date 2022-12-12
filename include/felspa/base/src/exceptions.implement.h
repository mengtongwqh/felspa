#ifndef _FELSPA_BASE_EXCEPTIONS_IMPLEMENT_H_
#define _FELSPA_BASE_EXCEPTIONS_IMPLEMENT_H_

#include <felspa/base/exceptions.h>
#include <felspa/base/utilities.h>

FELSPA_NAMESPACE_OPEN

/* --------------- */
namespace internal
/* --------------- */
{
  template <typename ExceptionType>
  [[noreturn]] void traced_throw(const ExceptionType& e)
  {
#ifdef FELSPA_HAS_BOOST_STACKTRACE
    throw boost::enable_error_info(e)
      << TracedErrorInfo(boost::stacktrace::stacktrace());
#else
    throw e;
#endif
  }


  template <typename ExceptionType>
  [[noreturn]] void throw_exception_with_tracing_info(ExceptionType exc,
                                                      const char* condition,
                                                      const int line,
                                                      const char* file,
                                                      const char* function)
  {
    if constexpr (std::is_base_of<ExceptionBase, ExceptionType>::value)
      exc.generate_errmsg(condition, FELSPA_DEMANGLE(exc), line, file,
                          function);
    traced_throw(exc);
  }


  template <typename ExceptionType>
  void print_exception_with_tracing_info(ExceptionType exc,
                                         const char* condition, const int line,
                                         const char* file, const char* function,
                                         std::ostream& ofile) noexcept
  {
    if constexpr (std::is_base_of<ExceptionBase, ExceptionType>::value)
      exc.generate_errmsg(condition, FELSPA_DEMANGLE(exc), line, file,
                          function);
    ofile << exc.what();

#ifdef FELSPA_HAS_BOOST_STACKTRACE
    ofile << "\n----- STACK TRACE -----\n";
    ofile << boost::stacktrace::stacktrace();
    ofile << "-----------------------\n" << std::endl;
#endif

    ofile << "====== END OF ERROR MESSAGE =====" << std::endl;
  }
}  // namespace internal


FELSPA_NAMESPACE_CLOSE

#endif  // _FELSPA_BASE_EXCEPTIONS_IMPLEMENT_H_ //