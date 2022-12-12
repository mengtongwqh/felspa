#ifndef _FELSPA_LEVEL_SET_VELOCITY_FIELD_H_
#define _FELSPA_LEVEL_SET_VELOCITY_FIELD_H_

#include <felspa/base/felspa_config.h>
#include <felspa/base/function.h>
#include <felspa/base/types.h>
#include <felspa/level_set/level_set.h>

#include <initializer_list>

FELSPA_NAMESPACE_OPEN

/* -------------------------------------------*/
namespace ls
/* -------------------------------------------*/
{
  /* ************************************************** */
  /**
   * \name Velocity field for Rigid Body Translation
   */
  /* ************************************************** */
  //@{

  template <int dim, typename NumberType = types::DoubleType>
  class RigidBodyTranslation : public TensorFunction<1, dim, NumberType>
  {
   public:
    using typename TensorFunction<1, dim, NumberType>::value_type;
    using typename TensorFunction<1, dim, NumberType>::point_type;
    using typename TensorFunction<1, dim, NumberType>::tensor_type;

    RigidBodyTranslation(std::initializer_list<value_type> velo);

    RigidBodyTranslation(const dealii::Tensor<1, dim, value_type>& velo);

    virtual tensor_type evaluate(const point_type&) const override
    {
      return velo_field;
    }

   protected:
    tensor_type velo_field;
  };

  //@}


  /* ************************************************** */
  /**
   * \name Velocity field for Rigid Body Rotation
   * \todo add definition for 3d rigid body rotation
   */
  /* ************************************************** */
  //@{
  template <int dim, typename NumberType = types::DoubleType>
  class RigidBodyRotation;  // useless generic declaration

  template <typename NumberType>
  class RigidBodyRotation<1, NumberType>
  {
   private:
    RigidBodyRotation() = default;
  };

  template <typename NumberType>
  class RigidBodyRotation<2, NumberType>
    : public TensorFunction<1, 2, NumberType>
  {
   public:
    using value_type = NumberType;
    using typename TensorFunction<1, 2, NumberType>::tensor_type;
    using typename TensorFunction<1, 2, NumberType>::point_type;

    // deafault the angular velocity to 2*PI/100 radian per second
    RigidBodyRotation(const point_type& center_,
                      value_type angular_velocity_ = constants::PI / 50.0);


    virtual tensor_type evaluate(const point_type& pt) const override;


   protected:
    /** Center of the velocity field, by default origin */
    point_type center;

    /** Angular velocity in radian, positive counterclockwise */
    value_type angular_velocity;
  };

  template <typename NumberType>
  class RigidBodyRotation<3, NumberType>
    : public TensorFunction<1, 3, NumberType>
  {
   public:
    RigidBodyRotation() { THROW(ExcNotImplemented()); }
  };
  //@}


  /* ************************************************** */
  /**
   * \name Velocity field for a single vortex flow.
   */
  /* ************************************************** */
  //@{

  template <int dim, typename NumberType = types::DoubleType>
  class SingleVortex;

  template <typename NumberType>
  class SingleVortex<1, NumberType> : public TensorFunction<1, 2, NumberType>
  {
   public:
    SingleVortex() { THROW(ExcNotImplemented()); }
  };

  template <typename NumberType>
  class SingleVortex<2, NumberType> : public TensorFunction<1, 2, NumberType>
  {
   public:
    using value_type = NumberType;
    using typename TensorFunction<1, 2, NumberType>::tensor_type;
    using typename TensorFunction<1, 2, NumberType>::point_type;

    SingleVortex(value_type period,
                 const point_type& center = {0.0, 0.0},
                 value_type scaling_coeff = constants::PI);

    tensor_type evaluate(const point_type& pt) const override;

   protected:
    /** Center of the velocity field */
    point_type center;

    /** Period of one time cycle */
    value_type T;

    /** Scaling coefficient */
    value_type scaling_coeff;
  };

  template <typename NumberType>
  class SingleVortex<3, NumberType> : public TensorFunction<3, 2, NumberType>
  {
   public:
    SingleVortex() { THROW(ExcNotImplemented()); }
  };

  //@}

}  // namespace ls
FELSPA_NAMESPACE_CLOSE
#endif
