#ifndef _FELSPA_BASE_EXCEPTION_CLASSES_H_
#define _FELSPA_BASE_EXCEPTION_CLASSES_H_

#include <felspa/base/exceptions.h>

FELSPA_NAMESPACE_OPEN

/* ********************************************** *
 * ######  STANDARDIZE COMMON EXCEPTIONS   ######
 * ********************************************** */

/* ************************************************** */
/**
 * \class ExcNotImplemented
 * \brief Aborting because the function requested has not been implemented
 * \ingroup Exception
 */
/* ************************************************** */
class ExcNotImplemented : public ExceptionBase
{
 public:
  ExcNotImplemented(const std::string& msg = "")
    : ExceptionBase(
        "This functionality requested has not been implemented in FELSPA "
        "library." +
        msg)
  {}
};


/* ************************************************** */
/**
 * \class ExcEmptyContainer
 * \brief Aborting because the container is expected to be nonempty
 * \ingroup Exception
 */
/* ************************************************** */
template <typename T>
class ExcEmptyContainer : public ExceptionBase
{
 public:
  ExcEmptyContainer(const std::string& msg = "")
    : ExceptionBase("The container cannot be empty. The container here is: " +
                    msg)
  {}
  virtual ~ExcEmptyContainer() = default;
};

#define ASSERT_NON_EMPTY(argm) \
  ASSERT((argm).size(),          \
         FELSPA_NAMESPACE::ExcEmptyContainer<decltype(argm)>(#argm))


/* ************************************************** */
/**
 * This class allows for printing of
 * unexpected value into the error message.
 */
/* ************************************************** */
template <typename T>
class ExcUnexpectedValue : public FELSPA_NAMESPACE::ExceptionBase
{
 public:
  ExcUnexpectedValue(T value) : arg(value) {}

 protected:
  virtual std::string specific_message() const override
  {
    std::ostringstream ss;
    ss << "Unexpected value encountered. The value = " << arg << '\n';
    return ss.str();
  }

  /// The stored unexpected value.
  T arg;
};


/* ************************************************** */
/**
 * The requested point is not on the boundary.
 */
/* ************************************************** */
template <int dim, typename NumberType = double>
class ExcPointNotOnBoundary : public ExceptionBase
{
 public:
  ExcPointNotOnBoundary(const dealii::Point<dim>& pt_)
    : ExceptionBase(), pt(pt_)
  {}

  virtual std::string specific_message() const override
  {
    std::ostringstream ss;
    ss << '(' << pt << ')' << " is not on the boundary\n";
    return ss.str();
  }

 protected:
  dealii::Point<dim, NumberType> pt;
};


/* ************************************************** */
/**
 * \class ExcSizeMismatch
 * \brief Aborting because the two containers are
 * expected to have the same size
 * \ingroup Exception
 */
/* ************************************************** */
DECL_EXCEPT_2(ExcSizeMismatch,
              "Containers are expected to have the same size! size(lhs) = "
                << arg1 << ", but size(rhs) = " << arg2 << '.',
              size_t, size_t);


#define ASSERT_SAME_SIZE(lhs, rhs)     \
  ASSERT((lhs).size() == (rhs).size(), \
         FELSPA_NAMESPACE::ExcSizeMismatch((lhs).size(), (rhs).size()))


/* ************************************************** */
/**
 * \class ExcInternalErr
 * \brief Throw this generic exception if
 * an internal check in the library has failed.
 * \ingroup Exception
 */
/* ************************************************** */
DECL_EXCEPT_0(ExcInternalErr, "FELSPA library failed an internal self-check.");


/* ************************************************** */
/**
 * \class ExcInternalErr
 * \brief Throw this generic exception if
 * an internal check in the library has failed.
 * \ingroup Exception
 */
/* ************************************************** */
DECL_EXCEPT_0(ExcArgumentCheckFail,
              "An argument passed to function has unexpected value");


/* ************************************************** */
/**
 * \class ExcNotImplementedInFileFormat
 * Printing the class in the current output format
 * has not been implemented
 * \ingroup Exception
 */
/* ************************************************** */
DECL_EXCEPT_1(ExcNotImplementedInFileFormat,
              "Executing the function in file format " << arg1
                                                       << " is not implemented",
              std::string);


/* ************************************************** */
/**
 * \class ExcDoFHandlerNotInit
 * \ingroup Exception
 */
/* ************************************************** */
DECL_EXCEPT_0(ExcDoFHandlerNotInit,
              "DoFHandler is not yet initialized and has no dofs. Call "
              "distribute_dofs() before this call.");


/* ************************************************** */
/**
 * \class ExcUnimplementedVirtualFcn
 * \ingroup Exception
 */
/* ************************************************** */
DECL_EXCEPT_0(ExcUnimplementedVirtualFcn,
              "This virtual function, "
              "defined in the base class, must be overridden in this class "
              "to produce meaningful result");


/* ************************************************** */
/**
 * \class ExcDividedByZero
 * \ingroup Exception
 */
/* ************************************************** */
DECL_EXCEPT_0(ExcDividedByZero, "The denominator is nearly zero.");


/* ************************************************** */
/**
 * \class ExcOutOfRange
 * \ingroup Exception
 */
/* ************************************************** */
DECL_EXCEPT_3(ExcOutOfRange,
              "Dereferencing index " << arg1 << " is not in the range of ["
                                     << arg2 << ',' << arg3 << ").",
              size_t, size_t, size_t);


/* ************************************************** */
/**
 * \class ExcNullPointer
 * \ingroup Exception
 */
/* ************************************************** */
DECL_EXCEPT_0(ExcNullPointer, "The pointer is not expected to be NULL pointer");


/* ************************************************** */
/**
 * \class ExcExpiredPointer
 * \ingroup Exception
 */
/* ************************************************** */
DECL_EXCEPT_0(ExcExpiredPointer, "The smart pointer has already expired.");


/* ************************************************** */
/**
 * \class ExcBackwardSync
 * \ingroup Exception
 */
/* ************************************************** */
DECL_EXCEPT_1(
  ExcBackwardSync,
  "Running the simulator backwards in time is not permissible. Time step = "
    << arg1,
  double);


/* ************************************************** */
/**
 * Expected an initialized simulator
 * \ingroup Exception
 */
/* ************************************************** */
DECL_EXCEPT_0(ExcSimulatorNotInitialized,
              "Simulator in an uninitialized state. Call initialize() before "
              "running the solver for problem solving.");


/* ************************************************** */
/**
 * Two triangulation is not the same.
 * \ingroup Exception
 */
/* ************************************************** */
DECL_EXCEPT_0(ExcMeshNotSame, "Must have the same Mesh.");


DECL_EXCEPT_0(ExcEmptyTriangulation, "The triangulation object is empty.");


DECL_EXCEPT_1(ExcWorksOnlyInSpaceDim,
              "The function only works in space dimension " << arg1, int);


DECL_EXCEPT_1(ExcNonExistentValueInSelection,
              "The selection string "
                << arg1 << " does not exist in the selection pattern.",
              std::string);


DECL_EXCEPT_2(ExcFEDegreeMismatch,
              "Two objects are expected to have the same finite element "
              "polynomial degree, but LHS degree = "
                << arg1 << " and RHS degree " << arg2 << '.',
              unsigned int, unsigned int);


DECL_EXCEPT_2(ExcMatrixNotSquare,
              "Matrix is not square: " << arg1 << 'x' << arg2, size_t, size_t);


DECL_EXCEPT_2(ExcMatrixVectorNotMultiplicable,
              "Matrix and vector is not multiplicable! Matrix has "
                << arg1 << " columns but the vector has " << arg2 << " entries",
              size_t, size_t);

#define ASSERT_MATRIX_VECTOR_MULTIPLICABLE(MATRIX, VECTOR) \
  ASSERT((MATRIX).n() == (VECTOR).size(),                  \
         ExcMatrixVectorNotMultiplicable((MATRIX).n(), (VECTOR).size()))


DECL_EXCEPT_2(ExcMatrixMatrixNotMultiplicable,
              "Matrix and matrix is not multiplicable! Left matrix has "
                << arg1 << " columns but the right matrix has " << arg2
                << " rows",
              size_t, size_t);

#define ASSERT_MATRIX_MATRIX_MULTIPLICABLE(LMATRIX, RMATRIX) \
  ASSERT((LMATRIX).n() == (RMATRIX).m(),                     \
         ExcMatrixMatrixNotMultiplicable((LMATRIX).n(), (RMATRIX).m()))


DECL_EXCEPT_0(ExcNotSynchronized,
              "The simulator and the object members are not synchronized.");


DECL_EXCEPT_0(ExcNotInitialized, "The object is not properly initialized.");


DECL_EXCEPT_0(ExcMeshUpdateUnprocessed,
              "Mesh update detected but not processed");


DECL_EXCEPT_0(ExcLinSysNotInit,
              "Linear System has not been initialized or populated. "
              "You should probably call populate_system_for_dofs()");


DECL_EXCEPT_2(ExcSolverFail,
              "Linear system solver reports non-convergence after "
                << arg1 << " with a tolerance of " << arg2,
              size_t, double);

#define ASSERT_SOLVER_CONVERGED(solver_control)                           \
  ASSERT((solver_control).last_check() == dealii::SolverControl::success, \
         ExcSolverFail((solver_control).last_step(),                      \
                       (solver_control).last_value()))


FELSPA_NAMESPACE_CLOSE

#endif  // _FELSPA_BASE_EXCEPTION_CLASSES_H_ //