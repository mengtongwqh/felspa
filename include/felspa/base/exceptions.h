#ifndef _FELSPA_BASE_EXCEPTIONS_H_
#define _FELSPA_BASE_EXCEPTIONS_H_

#include <deal.II/base/point.h>
#include <felspa/base/felspa_config.h>

#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef FELSPA_HAS_BOOST_STACKTRACE
#include <boost/exception/all.hpp>
#include <boost/stacktrace.hpp>
#endif  // FELSPA_HAS_BOOST_STACKTRACE //


/* ************************************************** */
/**
 * \defgroup Exception Exceptions, Assert and Error Handling
 * This module provides the error raising and handling mechanism
 * for FELSPA library.
 */
/* ************************************************** */
FELSPA_NAMESPACE_OPEN

/* -------------- */
namespace internal
/* -------------- */
{
#ifdef FELSPA_HAS_BOOST_STACKTRACE
  /* ************************************************** */
  /**
   * \brief Typedef for traced error in boost::stacktrace
   * \ingroup Exception
   */
  /* ************************************************** */
  using TracedErrorInfo =
    boost::error_info<struct tag_backtrace, boost::stacktrace::stacktrace>;
#endif  // #FELSPA_HAS_BOOST_STACKTRACE


  /* ************************************************** */
  /**
   * Setup the tracing information in the exception.
   * Then throw the exception with
   * \c boost::stacktrace injected.
   */
  /* ************************************************** */
  template <typename ExceptionType>
  [[noreturn]] void throw_exception_with_tracing_info(ExceptionType exc,
                                                      const char* condition,
                                                      const int line,
                                                      const char* file,
                                                      const char* function);


  /* ************************************************** */
  /**
   * Setup the tracing information in the exception.
   * Then print the error info with
   * \c boost::stacktrace injected.
   */
  /* ************************************************** */
  template <typename ExceptionType>
  void print_exception_with_tracing_info(
    ExceptionType exc, const char* condition, const int line, const char* file,
    const char* function, std::ostream& ofile = std::cerr) noexcept;

}  // namespace internal


/* ************************************************** */
/**
 * \brief Print exception info to stream object.
 * If boost::stacktrace is present,
 * also print the stacktrace of the exception.
 * \ingroup Exception
 * \tparam Exc exception type
 * \param[in] exc
 * \param[in, out] out stream object to write error info to
 */
/* ************************************************** */
void error_info(const std::exception& exc, std::ostream& out = std::cerr);


/* ************************************************** */
/**
 * \brief Print exception info to stream object.
 * If boost::stacktrace is present,
 * also print the stacktrace of the exception.
 * \ingroup Exception
 */
/* ************************************************** */
std::ostream& operator<<(std::ostream& out, const std::exception& exc);

/* ************************************************** */
/**
 * \class ExceptionBase
 * \brief Base class for all FELSPA exception classes
 * \ingroup Exception
 */
/* ************************************************** */
class ExceptionBase : public std::exception
{
 public:
  /**
   * @brief Constructor
   */
  ExceptionBase(const std::string& msg = "");

  /**
   * @brief Defaulted copy constructor
   */
  ExceptionBase(const ExceptionBase& rhs) = default;

  /**
   * @brief Virtual destructor
   */
  virtual ~ExceptionBase() = default;


  /**
   * \brief Overriding <code>what()</code> for
   * <code>std::exception</code> base class.
   * \return pointer to const char to errmsg
   */
  virtual const char* what() const noexcept override;


  /**
   * \brief Set all fields in the exception to describe where the exception
   * is thrown.
   * \param[in] line_ passed with __LINE__
   * \param[in] file_ passed with __FILE__
   * \param[in] function_ passed with __PRETTY_FUNCTION__
   */
  void generate_errmsg(const char* condition_, const std::string& name_,
                       const int line_, const char* file_,
                       const char* function_);

 protected:
  /**
   * \brief Populates errmsg field.
   * Called by <code>set_fields</code> to geneate a human-readable
   * error message to be used in <code>write_errmsg()</code> and
   * <code>what()</code> functions to display error message.
   */
  virtual std::string specific_message() const;


  /**
   * Violated condition which
   * causes exception to be thrown.
   */
  const char* condition;

  /**
   * Name of the exception class being thrown.
   */
  std::string name;

  /**
   * Line number in the file from
   * where exception is thrown
   */
  unsigned int line;


  /**
   * name of the file where exception is thrown
   */
  const char* file;

  /**
   * Name of the function from which the exception is thrown.
   */
  const char* function;

  /**
   * A brief error message,
   * passed into the constructor
   */
  std::string brief_msg;

  /**
   * Complete error message with
   * all tracing info generated.
   */
  mutable std::string errmsg;
};

FELSPA_NAMESPACE_CLOSE


/* ************************************************** */
/**
 * \brief Macro for testing asserted condition.
 * If the assertion failed,
 * exception \p exc will be thrown.
 * \ingroup Exception
 */
/* ************************************************** */
#ifdef DEBUG
#define ASSERT(cond, exc)                                          \
  if (!(cond))                                                     \
  ::FELSPA_NAMESPACE::internal::throw_exception_with_tracing_info( \
    (exc), #cond, __LINE__, __FILE__, __PRETTY_FUNCTION__)
#else
#define ASSERT(cond, exc) \
  {}
#endif  // DEBUG //


#ifdef DEBUG
#define ASSERT_WARN(cond, exc)                                     \
  if (!(cond))                                                     \
  ::FELSPA_NAMESPACE::internal::print_exception_with_tracing_info( \
    (exc), #cond, __LINE__, __FILE__, __PRETTY_FUNCTION__)
#else
#define ASSERT_WARN(cond, exc) \
  {}
#endif  // DEBUG //


#define THROW(exc)                                                 \
  ::FELSPA_NAMESPACE::internal::throw_exception_with_tracing_info( \
    (exc), "", __LINE__, __FILE__, __PRETTY_FUNCTION__)


/* ************************************************** */
/* The following set of macros are inspired by deal.II exception mechanism.
 * The idea is to declare the exception in class definition along with
 * the OutMsg and additional arguments you would like to pass to exception
 * class. The declared exception is derived from <code>ExceptionBase</code>
 * and thrown by ASSERT macro if cond is violated. The exception class is
 * constructed on the fly in ASSERT and thrown as an Rvalue.
 */
/* ************************************************** */
/**
 * \brief Construct a \c ExceptionBase object with input \c msg
 * \ingroup Exception
 */
#define EXCEPT_MSG(msg) FELSPA_NAMESPACE::ExceptionBase((msg))


/**
 * \brief Exceptions without further arguments
 * \ingroup Exception
 */
#define DECL_EXCEPT_0(ExcName, OutMsg)                    \
  class ExcName : public FELSPA_NAMESPACE::ExceptionBase  \
  {                                                       \
   protected:                                             \
    virtual std::string specific_message() const override \
    {                                                     \
      std::ostringstream ss;                              \
      ss << OutMsg;                                       \
      return ss.str();                                    \
    }                                                     \
  }


/**
 * \brief Exceptions with 1 argument
 * \ingroup Exception
 */
#define DECL_EXCEPT_1(ExcName, OutMsg, Type1)                             \
  class ExcName : public FELSPA_NAMESPACE::ExceptionBase                  \
  {                                                                       \
   public:                                                                \
    ExcName(Type1 arg1_)                                                  \
      : FELSPA_NAMESPACE::ExceptionBase(), Type1Name(#Type1), arg1(arg1_) \
    {}                                                                    \
                                                                          \
   protected:                                                             \
    virtual std::string specific_message() const override                 \
    {                                                                     \
      std::ostringstream ss;                                              \
      ss << OutMsg;                                                       \
      return ss.str();                                                    \
    }                                                                     \
    std::string Type1Name;                                                \
    Type1 arg1;                                                           \
  }


/**
 * \brief Exceptions with 1 defaulted argument
 * \ingroup Exception
 */
#define DECL_EXCEPT_DEFAULT_1(ExcName, OutMsg, Type1, DefArg1)            \
  class ExcName : public FELSPA_NAMESPACE::ExceptionBase                  \
  {                                                                       \
   public:                                                                \
    ExcName(Type1 arg1_ = DefArg1)                                        \
      : FELSPA_NAMESPACE::ExceptionBase(), Type1Name(#Type1), arg1(arg1_) \
    {}                                                                    \
                                                                          \
   protected:                                                             \
    virtual std::string specific_message() const override                 \
    {                                                                     \
      std::ostringstream ss;                                              \
      ss << OutMsg;                                                       \
      return ss.str();                                                    \
    }                                                                     \
    std::string Type1Name;                                                \
    Type1 arg1;                                                           \
  }


/**
 * \brief Exceptions with 2 arguments
 * \ingroup Exception
 */
#define DECL_EXCEPT_2(ExcName, OutMsg, Type1, Type2)      \
  class ExcName : public FELSPA_NAMESPACE::ExceptionBase  \
  {                                                       \
   public:                                                \
    ExcName(Type1 arg1_, Type2 arg2_)                     \
      : FELSPA_NAMESPACE::ExceptionBase(),                \
        Type1Name(#Type1),                                \
        Type2Name(#Type2),                                \
        arg1(arg1_),                                      \
        arg2(arg2_)                                       \
    {}                                                    \
                                                          \
   protected:                                             \
    virtual std::string specific_message() const override \
    {                                                     \
      std::ostringstream ss;                              \
      ss << OutMsg;                                       \
      return ss.str();                                    \
    }                                                     \
    std::string Type1Name;                                \
    std::string Type2Name;                                \
    Type1 arg1;                                           \
    Type2 arg2;                                           \
  }


/**
 * \brief Exceptions with 2 defaulted arguments
 * \ingroup Exception
 */
#define DECL_EXCEPT_DEFAULT_2(ExcName, OutMsg, Type1, Type2, DefArg1, DefArg2) \
  class ExcName : public FELSPA_NAMESPACE::ExceptionBase                       \
  {                                                                            \
   public:                                                                     \
    ExcName(Type1 arg1_ = DefArg1, Type2 arg2_ = DefArg2)                      \
      : FELSPA_NAMESPACE::ExceptionBase(),                                     \
        Type1Name(#Type1),                                                     \
        Type2Name(#Type2),                                                     \
        arg1(arg1_),                                                           \
        arg2(arg2_)                                                            \
    {}                                                                         \
                                                                               \
   protected:                                                                  \
    virtual std::string specific_message() const override                      \
    {                                                                          \
      std::ostringstream ss;                                                   \
      ss << OutMsg;                                                            \
      return ss.str();                                                         \
    }                                                                          \
    std::string Type1Name;                                                     \
    std::string Type2Name;                                                     \
    Type1 arg1;                                                                \
    Type2 arg2;                                                                \
  }


/**
 * \brief Exceptions with 3 arguments
 * \ingroup Exception
 */
#define DECL_EXCEPT_3(ExcName, OutMsg, Type1, Type2, Type3) \
  class ExcName : public FELSPA_NAMESPACE::ExceptionBase    \
  {                                                         \
   public:                                                  \
    ExcName(Type1 arg1_, Type2 arg2_, Type3 arg3_)          \
      : FELSPA_NAMESPACE::ExceptionBase(),                  \
        Type1Name(#Type1),                                  \
        Type2Name(#Type2),                                  \
        Type3Name(#Type3),                                  \
        arg1(arg1_),                                        \
        arg2(arg2_),                                        \
        arg3(arg3_)                                         \
    {}                                                      \
                                                            \
   protected:                                               \
    virtual std::string specific_message() const override   \
    {                                                       \
      std::ostringstream ss;                                \
      ss << OutMsg;                                         \
      return ss.str();                                      \
    }                                                       \
    std::string Type1Name;                                  \
    std::string Type2Name;                                  \
    std::string Type3Name;                                  \
    Type1 arg1;                                             \
    Type2 arg2;                                             \
    Type3 arg3;                                             \
  }


/**
 * \brief Exceptions with 3 defaulted arguments
 * \ingroup Exception
 */
#define DECL_EXCEPT_DEFAULT_3(ExcName, OutMsg, Type1, Type2, Type3, DefArg1, \
                              DefArg2, DefArg3)                              \
  class ExcName : public FELSPA_NAMESPACE::ExceptionBase                     \
  {                                                                          \
   public:                                                                   \
    ExcName(Type1 arg1_ = DefArg1, Type2 arg2_ = DefArg2,                    \
            Type3 arg3_ = DefArg3)                                           \
      : FELSPA_NAMESPACE::ExceptionBase(),                                   \
        Type1Name(#Type1),                                                   \
        Type2Name(#Type2),                                                   \
        arg1(arg1_),                                                         \
        arg2(arg2_),                                                         \
        arg3(arg3_)                                                          \
    {}                                                                       \
                                                                             \
   protected:                                                                \
    virtual std::string specific_message() const override                    \
    {                                                                        \
      std::ostringstream ss;                                                 \
      ss << OutMsg;                                                          \
      return ss.str();                                                       \
    }                                                                        \
    std::string Type1Name;                                                   \
    std::string Type2Name;                                                   \
    std::string Type3Name;                                                   \
    Type1 arg1;                                                              \
    Type2 arg2;                                                              \
    Type3 arg3;                                                              \
  }


/**
 * \brief Exceptions with 4 arguments
 * \ingroup Exception
 */
#define DECL_EXCEPT_4(ExcName, OutMsg, Type1, Type2, Type3, Type4) \
  class ExcName : public FELSPA_NAMESPACE::ExceptionBase           \
  {                                                                \
   public:                                                         \
    ExcName(Type1 arg1_, Type2 arg2_, Type3 arg3_, Type4 arg4_)    \
      : FELSPA_NAMESPACE::ExceptionBase(),                         \
        Type1Name(#Type1),                                         \
        Type2Name(#Type2),                                         \
        Type3Name(#Type3),                                         \
        Type4Name(#Type4),                                         \
        arg1(arg1_),                                               \
        arg2(arg2_),                                               \
        arg3(arg3_),                                               \
        arg4(arg4_)                                                \
    {}                                                             \
                                                                   \
   protected:                                                      \
    virtual std::string specific_message() const override          \
    {                                                              \
      std::ostringstream ss;                                       \
      ss << OutMsg;                                                \
      return ss.str();                                             \
    }                                                              \
    std::string Type1Name;                                         \
    std::string Type2Name;                                         \
    std::string Type3Name;                                         \
    std::string Type4Name;                                         \
    Type1 arg1;                                                    \
    Type2 arg2;                                                    \
    Type3 arg3;                                                    \
    Type4 arg4;                                                    \
  }


#include "src/exceptions.implement.h"
#include "src/exception_classes.h" // collections of exceptions

#endif  // #ifndef _FELSPA_BASE_EXCEPTIONS_H_
