
#include <felspa/base/exceptions.h>

FELSPA_NAMESPACE_OPEN

/* --------------------------------------------------*/
/** \class ExceptionBase */
/* --------------------------------------------------*/
ExceptionBase::ExceptionBase(const std::string& what_arg)
  : condition(nullptr),
    name(""),
    line(0),
    file(nullptr),
    function(nullptr),
    brief_msg(what_arg)
{}


FELSPA_FORCE_INLINE
std::string ExceptionBase::specific_message() const
{
  return "<No additional info>";
}


const char* ExceptionBase::what() const noexcept
{
  if (errmsg.empty()) {
    errmsg = "\n========= FELSPA Library exception =========\n";
    errmsg += "<Exception info is not recorded>";
    errmsg += "Error message: " + brief_msg + '\n';
    errmsg += specific_message() + '\n';
    errmsg += "====== END OF ERROR MESSAGE =====\n";
  }
  return errmsg.c_str();
}


void ExceptionBase::generate_errmsg(const char* condition_,
                                    const std::string& name_, const int line_,
                                    const char* file_, const char* function_)
{
  errmsg.clear();

  condition = condition_;
  name = name_;
  line = static_cast<unsigned int>(line_);
  file = file_;
  function = function_;

  std::ostringstream ss;
  ss << "\n========== FELSPA Library Exception ========= \n";
  ss << "Name of the Exception: " << name << '\n';
  ss << "Raised due to the violation of the condition: (" << condition << ")\n";
  ss << "Thrown from function: " << function << '\n';
  ss << "In line [" << line << "] of file [" << file << "]\n";
  ss << "Error message: " << brief_msg << '\n';
  ss << specific_message() << '\n';
  errmsg = ss.str();
}


void error_info(const std::exception& exc, std::ostream& out)
{
  // print the generated errmsg with tracing info
  out << exc.what();

#ifdef FELSPA_HAS_BOOST_STACKTRACE
  // if the boost stacktrace has been injected
  // we are able to retrive that information here.
  using boost::get_error_info;
  using boost::stacktrace::stacktrace;
  using internal::TracedErrorInfo;

  out << "\n----- STACK TRACE -----\n";
  const stacktrace* st = get_error_info<TracedErrorInfo>(exc);
  if (st)
    out << *st;
  else
    out << "<Empty stacktrace>\n";
  out << "-----------------------\n" << std::endl;
#endif  // FELSPA_HAS_BOOST_STACKTRACE

  out << "====== END OF ERROR MESSAGE =====" << std::endl;
}


std::ostream& operator<<(std::ostream& out, const std::exception& exc)
{
  error_info(exc, out);
  return out;
}


FELSPA_NAMESPACE_CLOSE
