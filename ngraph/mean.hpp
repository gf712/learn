#include "ngraph/op/op.hpp"
#include "ngraph/op/util/arithmetic_reduction.hpp"
#include "ngraph/runtime/cpu/builder/reduction.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/reference/sum.hpp"
#include <unsupported/Eigen/CXX11/Tensor>

namespace ngraph
{
	namespace op
	{
		class MeanOp : public util::ArithmeticReduction
		{
		public:
			static constexpr NodeTypeInfo type_info{"MeanOp", 0};
			const NodeTypeInfo& get_type_info() const override
			{
				return type_info;
			}
			/// \brief Constructs a MeanOp operation.
			MeanOp() = default;
			/// \brief Constructs a MeanOp operation.
			///
			/// \param arg The tensor to be averaged.
			/// \param reduction_axes The axis positions (0-based) to be
			/// eliminated.
			MeanOp(const Output<Node>& arg, const AxisSet& reduction_axes);
			/// \brief Constructs a MeanOpoperation.
			///
			/// \param arg The tensor to be MeanOp.
			/// \param reduction_axes The axis positions (0-based) to be
			/// eliminated.
			MeanOp(const Output<Node>& arg, const Output<Node>& reduction_axes);

			virtual std::shared_ptr<Node>
			copy_with_new_args(const NodeVector& new_args) const override;

			/// \return The default value for MeanOp.
			virtual std::shared_ptr<Node> get_default_value() const override;

		protected:
			virtual void generate_adjoints(
			    autodiff::Adjoints& adjoints,
			    const OutputVector& deltas) override;
		};
	} // namespace op
} // namespace ngraph

namespace ngraph
{
	namespace runtime
	{
		namespace reference
		{
			template <typename T>
			void mean(
			    const T* arg, T* out, const Shape& in_shape,
			    const Shape& out_shape, const AxisSet& reduction_axes)
			{
				CoordinateTransform output_transform(out_shape);
				std::vector<T> cs(shape_size(out_shape));
				std::vector<T> counter(shape_size(out_shape));

				for (const Coordinate& output_coord : output_transform)
				{
					out[output_transform.index(output_coord)] = 0;
					cs[output_transform.index(output_coord)] = 0;
					counter[output_transform.index(output_coord)] = 0;
				}

				CoordinateTransform input_transform(in_shape);

				for (const Coordinate& input_coord : input_transform)
				{
					Coordinate output_coord =
					    reduce(input_coord, reduction_axes);

					T x = arg[input_transform.index(input_coord)];
					T& z = out[output_transform.index(output_coord)];
					T& counter_i =
					    counter[output_transform.index(output_coord)];

					if (is_finite(x) && is_finite(z))
					{
						T& c = cs[output_transform.index(output_coord)];
						T t = z + (x - c);
						c = (t - z) - (x - c);
						z = t;
					}
					else
					{
						z += x;
					}
					counter_i++;
				}

				for (const Coordinate& output_coord : output_transform)
					out[output_transform.index(output_coord)] /=
					    counter[output_transform.index(output_coord)];
			}
		} // namespace reference
	}     // namespace runtime
} // namespace ngraph

namespace ngraph
{
	namespace runtime
	{
		namespace cpu
		{
			namespace kernel
			{
				template <typename ElementType, unsigned int Rank>
				void reduce_mean_all(
				    void* input, void* output, const Shape& input_shape,
				    const Shape& /* output_shape */, int arena)
				{
					Eigen::array<Eigen::Index, Rank> in_dims;
					Eigen::array<Eigen::Index, 0> out_dims;

					for (size_t i = 0; i < Rank; i++)
					{
						in_dims[i] = input_shape[i];
					}

					Eigen::TensorMap<
					    Eigen::Tensor<ElementType, 0, Eigen::RowMajor>>
					out(static_cast<ElementType*>(output), out_dims);
					Eigen::TensorMap<
					    Eigen::Tensor<ElementType, Rank, Eigen::RowMajor>>
					in(static_cast<ElementType*>(input), in_dims);
					out.device(ngraph::runtime::cpu::executor::GetCPUExecutor()
					               .get_device(arena)) = in.mean();
				}
				template <typename ElementType, unsigned int Rank>
				void reduce_mean_innermost_1rd(
				    void* input, void* output, const Shape& input_shape,
				    const Shape& output_shape, int arena)
				{
					Eigen::array<Eigen::Index, Rank> in_dims;
					Eigen::array<Eigen::Index, Rank - 1> out_dims;
					Eigen::IndexList<Eigen::type2index<Rank - 1>> reduction_dim;

					for (size_t i = 0; i < Rank; i++)
					{
						in_dims[i] = input_shape[i];
					}

					for (size_t i = 0; i < Rank - 1; i++)
					{
						out_dims[i] = output_shape[i];
					}

					Eigen::TensorMap<
					    Eigen::Tensor<ElementType, Rank - 1, Eigen::RowMajor>>
					out(static_cast<ElementType*>(output), out_dims);
					Eigen::TensorMap<
					    Eigen::Tensor<ElementType, Rank, Eigen::RowMajor>>
					in(static_cast<ElementType*>(input), in_dims);
					out.device(ngraph::runtime::cpu::executor::GetCPUExecutor()
					               .get_device(arena)) = in.mean(reduction_dim);
				}

				template <
				    typename ElementType, unsigned int Rank,
				    unsigned int ReductionDims>
				void reduce_mean(
				    void* input, void* output, const Shape& input_shape,
				    const Shape& output_shape, const AxisSet& reduction_axes,
				    int arena)
				{
					Eigen::array<Eigen::Index, Rank> in_dims;
					Eigen::array<Eigen::Index, Rank - ReductionDims> out_dims;
					Eigen::array<Eigen::Index, ReductionDims> reduction_dims;

					for (size_t i = 0; i < Rank; i++)
					{
						in_dims[i] = input_shape[i];
					}

					for (size_t i = 0; i < Rank - ReductionDims; i++)
					{
						out_dims[i] = output_shape[i];
					}

					size_t i = 0;
					for (auto axis : reduction_axes)
					{
						reduction_dims[i++] = axis;
					}

					Eigen::TensorMap<Eigen::Tensor<
					    ElementType, Rank - ReductionDims, Eigen::RowMajor>>
					out(static_cast<ElementType*>(output), out_dims);
					Eigen::TensorMap<
					    Eigen::Tensor<ElementType, Rank, Eigen::RowMajor>>
					in(static_cast<ElementType*>(input), in_dims);
					out.device(ngraph::runtime::cpu::executor::GetCPUExecutor()
					               .get_device(arena)) =
					    in.mean(reduction_dims);
				}

				template <typename ElementType, unsigned int Rank>
				void reduce_mean_1rd(
				    void* input, void* output, const Shape& input_shape,
				    const Shape& output_shape, const AxisSet& reduction_axes,
				    int arena)
				{
					reduce_mean<ElementType, Rank, 1>(
					    input, output, input_shape, output_shape,
					    reduction_axes, arena);
				}

				template <typename ElementType>
				void reduce_mean_3d_2rd(
				    void* input, void* output, const Shape& input_shape,
				    const Shape& output_shape, const AxisSet& reduction_axes,
				    int arena)
				{
					reduce_mean<ElementType, 3, 2>(
					    input, output, input_shape, output_shape,
					    reduction_axes, arena);
				}

				template <typename ElementType>
				void reduce_mean_4d_2rd(
				    void* input, void* output, const Shape& input_shape,
				    const Shape& output_shape, const AxisSet& reduction_axes,
				    int arena)
				{
					reduce_mean<ElementType, 4, 2>(
					    input, output, input_shape, output_shape,
					    reduction_axes, arena);
				}

				template <typename ElementType>
				void reduce_mean_5d_2rd(
				    void* input, void* output, const Shape& input_shape,
				    const Shape& output_shape, const AxisSet& reduction_axes,
				    int arena)
				{
					reduce_mean<ElementType, 5, 2>(
					    input, output, input_shape, output_shape,
					    reduction_axes, arena);
				}

				template <typename ElementType>
				void mean(
				    void* arg, void* out, const Shape& in_shape,
				    const Shape& out_shape, const AxisSet& reduction_axes,
				    int /* arena */)
				{
					reference::mean(
					    static_cast<ElementType*>(arg),
					    static_cast<ElementType*>(out), in_shape, out_shape,
					    reduction_axes);
				}

			} // namespace kernel
		}     // namespace cpu
	}         // namespace runtime
} // namespace ngraph