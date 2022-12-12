#ifndef _FELSPA_LEVEL_SET_GEOMETRY_H_
#define _FELSPA_LEVEL_SET_GEOMETRY_H_

#include <deal.II/base/point.h>
#include <deal.II/base/smartpointer.h>
#include <felspa/base/felspa_config.h>
#include <felspa/base/function.h>
#include <felspa/base/numerics.h>
#include <felspa/base/types.h>

FELSPA_NAMESPACE_OPEN

/* ------------------------------ */
namespace ls
/* ------------------------------ */
{
  /* ************************************************** */
  /** \name Initial Condition Base and Shape Operations */
  /* ************************************************** */
  //@{
  /**
   * \class ICBase
   * Base class for level set initial condition
   * \tparam dim
   * \tparam NumberType
   */
  template <int dim, typename NumberType = types::DoubleType>
  class ICBase : public ScalarFunction<dim, NumberType>
  {
   public:
    using value_type = NumberType;
    using typename ScalarFunction<dim, NumberType>::point_type;

    /** \c true if the level set is exactly/analytically computed */
    const bool exact;

   protected:
    /** Hidden constructor */
    ICBase(bool level_set_is_exact) : exact(level_set_is_exact) {}
  };


  /**
   * \class ICBinaryOp
   * Base class for binary operators of level set initial conditions
   * \tparam dim
   * \tparam NumberType
   */
  template <int dim, typename NumberType = types::DoubleType>
  class ICBinaryOp : public ICBase<dim, NumberType>
  {
   public:
    using value_type = NumberType;
    using typename ScalarFunction<dim, NumberType>::point_type;

   protected:
    /** Hidden constructor */
    ICBinaryOp(const ICBase<dim, NumberType>& ic1,
               const ICBase<dim, NumberType>& ic2)
      : ICBase<dim, NumberType>(ic1.exact && ic2.exact),
        ptr_ic1(&ic1),
        ptr_ic2(&ic2)
    {}

    /** (Pointer to) 1st operand */
    dealii::SmartPointer<const ICBase<dim, NumberType>> ptr_ic1;

    /** (Pointer to) 2nd operand */
    dealii::SmartPointer<const ICBase<dim, NumberType>> ptr_ic2;
  };


  /**
   * \class ICUnion
   * Compute the union of two level set initial conditions
   * \tparam dim
   * \tparam NumberType
   */
  template <int dim, typename NumberType = types::DoubleType>
  class ICUnion : public ICBinaryOp<dim, NumberType>
  {
   public:
    using value_type = NumberType;
    using typename ICBase<dim, NumberType>::point_type;

    ICUnion(const ICBase<dim, NumberType>& ic1,
            const ICBase<dim, NumberType>& ic2)
      : ICBinaryOp<dim, NumberType>(ic1, ic2)
    {}

    value_type evaluate(const point_type& pt) const override
    {
      return std::min((*this->ptr_ic1)(pt), (*this->ptr_ic2)(pt));
    }
  };


  template <int dim, typename NumberType = types::DoubleType>
  class ICDiff : public ICBinaryOp<dim, NumberType>
  {
   public:
    using value_type = NumberType;
    using typename ICBase<dim, NumberType>::point_type;

    ICDiff(const ICBase<dim, NumberType>& ic1,
           const ICBase<dim, NumberType>& ic2)
      : ICBinaryOp<dim, NumberType>(ic1, ic2)
    {}

    value_type evaluate(const point_type& pt) const override
    {
      return std::max((*this->ptr_ic1)(pt), -(*this->ptr_ic2)(pt));
    }
  };


  template <int dim, typename NumberType = types::DoubleType>
  class ICIntersect : public ICBinaryOp<dim, NumberType>
  {
   public:
    using value_type = NumberType;
    using typename ICBase<dim, NumberType>::point_type;

    ICIntersect(const ICBase<dim, NumberType>& ic1,
                const ICBase<dim, NumberType>& ic2)
      : ICBinaryOp<dim, NumberType>(ic1, ic2)
    {}
    value_type evaluate(const point_type& pt) const override
    {
      return std::max((*this->ptr_ic1)(pt), (*this->ptr_ic2)(pt));
    }
  };
  //@}

  /* ************************************************** */
  /** \name Operator overloading for shape operations   */
  /* ************************************************** */
  //@{
  template <int dim, typename NumberType>
  ICUnion<dim, NumberType> operator+(const ICBase<dim, NumberType>& ic1,
                                     const ICBase<dim, NumberType>& ic2)
  {
    return ICUnion<dim, NumberType>(ic1, ic2);
  }

  template <int dim, typename NumberType>
  ICDiff<dim, NumberType> operator-(const ICBase<dim, NumberType>& ic1,
                                    const ICBase<dim, NumberType>& ic2)
  {
    return ICDiff<dim, NumberType>(ic1, ic2);
  }

  template <int dim, typename NumberType>
  ICIntersect<dim, NumberType> operator*(const ICBase<dim, NumberType>& ic1,
                                         const ICBase<dim, NumberType>& ic2)
  {
    return ICIntersect<dim, NumberType>(ic1, ic2);
  }
  //@}


  /* ************************************************** */
  /**
   * \brief HyperSphere
   * For constructing a dimension-independent spherical geometry
   * \tparam dim
   * \tparam NumberType
   */
  /* ************************************************** */
  template <int dim, typename NumberType = types::DoubleType>
  class HyperSphere : public ICBase<dim, NumberType>
  {
   public:
    using typename ICBase<dim, NumberType>::point_type;
    using value_type = NumberType;

    /**
     * \brief HyperSphere Constructor
     * \param[in] center The center of the sphere
     * \param[in] radius The radius of the sphere
     */
    HyperSphere(const point_type& center_, value_type radius_);

    virtual value_type evaluate(const point_type&) const override;

   protected:
    /** Center of the sphere */
    dealii::Point<dim> center;

    /** Radius of the sphere */
    value_type radius;
  };  // class HyperSphere


  /* ************************************************** */
  /**
   * @brief  HyperPlane
   * The sign is determined by the inner product with
   * the normal vector.
   * For 2d, normal vector is on
   * the left side of the tangent vector. \n
   * For 3d, normal vector is the result of t1 x t2, where t1,t2
   * are tangent vectors.
   */
  /* ************************************************** */
  template <int dim, typename NumberType>
  class HyperPlaneBase : public ICBase<dim, NumberType>

  {
   public:
    using value_type = NumberType;
    using point_type = dealii::Point<dim, NumberType>;

    value_type evaluate(const point_type& pt) const override;

   protected:
    /**
     * Constructor
     */
    HyperPlaneBase(const point_type& pt)
      : ICBase<dim, value_type>(true), ref_point(pt)
    {}

    /**
     * @brief Reference point.
     * The surface will be anchored at the reference point.
     */
    point_type ref_point;

    /**
     * @brief  Normalized tangent vector
     */
    dealii::Tensor<1, dim, value_type> normal_vector;
  };


  template <int dim, typename NumberType = types::DoubleType>
  class HyperPlane;


  template <typename NumberType>
  class HyperPlane<2, NumberType> : public HyperPlaneBase<2, NumberType>
  {
   public:
    constexpr static int dimension = 2, dim = dimension;
    using value_type = NumberType;
    using point_type = dealii::Point<dim, value_type>;


    /**
     * @brief Constructor
     * @param pt  the reference point
     * @param tangent_vector  tangent vector at the reference point
     */
    HyperPlane(const point_type& pt,
               const dealii::Tensor<1, dim, value_type>& tangent_vector);
  };


  template <typename NumberType>
  class HyperPlane<3, NumberType> : public HyperPlaneBase<3, NumberType>
  {
   public:
    constexpr static int dimension = 3, dim = dimension;

    using value_type = NumberType;
    using point_type = dealii::Point<dim, value_type>;


    HyperPlane(const point_type& pt,
               const dealii::Tensor<1, dim, value_type>& t1,
               const dealii::Tensor<1, dim, value_type>& t2);
  };


  /* ************************************************** */
  /** \name Level set for rectangle */
  /* ************************************************** */
  //@{
  template <int dim, typename NumberType = types::DoubleType>
  class HyperRectangleBase : public ICBase<dim, NumberType>
  {
   public:
    using typename ICBase<dim, NumberType>::point_type;

    using value_type = NumberType;

   protected:
    /** Constructor */
    HyperRectangleBase(const point_type& pt1_, const point_type& pt2_)
      : ICBase<dim, NumberType>(true), pt1(pt1_), pt2(pt2_)
    {
      for (int idim = 0; idim < dim; ++idim) {
        ASSERT(
          pt1(idim) < pt2(idim),
          EXCEPT_MSG("pt1 coordinate must be less than pt2 in each direction"));
      }
    }

    /** lower left point on the diagonal */
    point_type pt1;
    /** upper right point on the diagonal */
    point_type pt2;
  };  // class HyperRectangleBase<dim, NumberType>

  template <int dim, typename NumberType = types::DoubleType>
  class HyperRectangle : public HyperRectangleBase<dim, NumberType>
  {
   public:
    using typename ICBase<1, NumberType>::point_type;
    using value_type = NumberType;

   protected:
    HyperRectangle(const point_type& pt1, const point_type& pt2);
  };  // class HyperRectangle<dim, NumberType>

  template <typename NumberType>
  class HyperRectangle<1, NumberType> : public HyperRectangleBase<1, NumberType>
  {
   public:
    using typename ICBase<1, NumberType>::point_type;
    using value_type = NumberType;

    HyperRectangle(const point_type& pt1_, const point_type& pt2_)
      : HyperRectangleBase<1, NumberType>(pt1_, pt2_)
    {}

    virtual value_type evaluate(const point_type&) const override;
  };  // class HyperRectangle<1, NumberType>

  template <typename NumberType>
  class HyperRectangle<2, NumberType> : public HyperRectangleBase<2, NumberType>
  {
   public:
    using typename HyperRectangleBase<2, NumberType>::point_type;
    using value_type = NumberType;

    HyperRectangle(const point_type& pt1_, const point_type& pt2_)
      : HyperRectangleBase<2, NumberType>(pt1_, pt2_)
    {}

    virtual value_type evaluate(const point_type&) const override;
  };  // class HyperRectangle<2, NumberType>

  template <typename NumberType>
  class HyperRectangle<3, NumberType> : public HyperRectangleBase<3, NumberType>
  {
   public:
    using typename ICBase<3, NumberType>::point_type;
    using value_type = NumberType;

    HyperRectangle(const point_type& pt1_, const point_type& pt2_)
      : HyperRectangleBase<3, NumberType>(pt1_, pt2_)
    {}

    virtual value_type evaluate(const point_type&) const override;
  };  // class HyperRectangle<3, NumberType>
  //@}


  /* ************************************************** */
  /**
   * \brief Step initial condition
   * Construct a discontinuous step of given width and
   * height centerd at a \p center point.
   * This is not an initial condition that corresponds to
   * any level set geometry.
   * We include this function to
   * test the performance of our
   * discretization for discontinuous function.
   */
  /* ************************************************** */
  template <int dim, typename NumberType = types::DoubleType>
  class Step : public ICBase<dim, NumberType>
  {
   public:
    using value_type = NumberType;
    using typename ICBase<dim, NumberType>::point_type;

    Step(const point_type& center_, value_type width_,
         value_type height_ = 1.0);

    virtual value_type evaluate(const point_type&) const override;

   protected:
    point_type center;
    value_type width;
    value_type height;
  };  // class Step


  /* ************************************************** */
  /**
   * \brief Initial condition for consine cone
   */
  /* ************************************************** */
  template <int dim, typename NumberType = types::DoubleType>
  class CosineCone : public ICBase<dim, NumberType>
  {
   public:
    using value_type = NumberType;
    using typename ICBase<dim, NumberType>::point_type;
    CosineCone(const point_type& center_, value_type radius_);

    virtual value_type evaluate(const point_type&) const override;

   protected:
    point_type center;
    value_type radius;
  };


  /* ************************************************** */
  /**
   * Initial condition for a perturbed circle.
   * This geometry is useful for testing reinitialization
   */
  /* ************************************************** */
  template <int dim, typename NumberType = types::DoubleType>
  class SmoothPerturbedSphere : public ICBase<dim, NumberType>
  {
   public:
    using value_type = NumberType;

    using typename ICBase<dim, NumberType>::point_type;

    SmoothPerturbedSphere(const point_type center = {0.0, 0.0},
                          value_type radius = 1.0,
                          value_type perturb_coeff = 0.1,
                          point_type ref_point = {1.0, 1.0});

    void set_reference_point(std::initializer_list<value_type>);

    virtual value_type evaluate(const point_type&) const override;

   protected:
    point_type center;

    value_type radius;

    value_type perturb_coeff;

    point_type reference_point;
  };


  /* ************************************************** */
  /**
   * Consine hill in multidimensions.
   */
  /* ************************************************** */
  template <int dim, typename NumberType = types::DoubleType>
  class CosineTensorProduct : public ICBase<dim, NumberType>
  {
   public:
    using value_type = NumberType;
    using typename ICBase<dim, NumberType>::point_type;

    CosineTensorProduct(const point_type& center,
                        const point_type& period,
                        value_type amplitude = 1.0);

    virtual value_type evaluate(const point_type& pt) const override;

   protected:
    point_type center;

    point_type period;

    value_type amplitude;
  };


  /* ************************************************** */
  /**
   * Rayleigh Taylor instability initial condition. \n
   * - A - amplitude \n
   * - \f$(x_0, y_0, z_0)\f$ - center point \n
   * - \f$\lambda\f$ - period \n
   *
   * The surface in 2d is:
   * \f[
   * f(x) =  A\cos{(2\pi\lambda(x - x_0))} + y_0
   * constrained to \f$ [-\lambda, \lambda] \f$
   * \f]
   *
   * The surface in 3d, if we do tensor product, is:
   * \f[
   * f(x,y) = A\cos{(2\pi\lambda_x(x - x_0))}\cos{(2\pi\lambda_y(y - y_0))} +
   * z_0
   * constrained to  \f$ |\mathbf(x)| \in [-\lambda, \lambda]
   * \f]
   *
   * The surface in 3d, if we do
   */
  /* ************************************************** */
  template <int dim, typename NumberType = types::DoubleType>
  class RayleighTaylorLower : public ICBase<dim, NumberType>
  {
   public:
    using value_type = NumberType;
    using typename ICBase<dim, NumberType>::point_type;
    static constexpr int dimension = dim;


    /**
     * Constructor
     */
    RayleighTaylorLower(const point_type& pt, value_type period,
                        value_type amplitude);


    /**
     * We cannot compute a closed form solution to this initial condition,
     * so compute a (linearized) derivative approximation.
     */
    value_type evaluate(const point_type& pt) const override;


   protected:
    /**
     * Expresssion of the level set surface
     */
    value_type surface_fcn(const point_type&) const;

    point_type center;

    value_type period;

    value_type amplitude;
  };
}  // namespace ls

FELSPA_NAMESPACE_CLOSE
#endif  // _FELSPA_LEVEL_SET_GEOMETRY_H_
