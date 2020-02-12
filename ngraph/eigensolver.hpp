#include "ngraph/op/op.hpp"

#include "ngraph/runtime/cpu/cpu_builder.hpp"

#include <Eigen/SVD>
#include <unsupported/Eigen/CXX11/Tensor>

namespace ngraph
{
	namespace op
	{
		class JacobiSVD : public Op
		{
		public:
			static constexpr NodeTypeInfo type_info{"JacobiSVD", 0};

			const NodeTypeInfo& get_type_info() const override
			{
				return type_info;
			}

			JacobiSVD() = default;

			JacobiSVD(const Output<Node>& input0);

			void validate_and_infer_types() override;

			std::shared_ptr<Node>
			copy_with_new_args(const NodeVector& new_args) const override;

			bool visit_attributes(AttributeVisitor& visitor) override;
		};
	} // namespace op
} // namespace ngraph

namespace ngraph
{
	namespace runtime
	{
		namespace cpu
		{
			namespace kernel
			{
				template <typename ElementType>
				void jacobi_svd(
				    void* input0, void* output1, void* output2,
				    const Shape& input1_shape, const Shape& output1_shape,
				    const Shape& output2_shape, int arena)
				{
					using MatrixType = Eigen::Matrix<
					    ElementType, Eigen::Dynamic, Eigen::Dynamic>;

					Eigen::array<Eigen::Index, 2> in_dims{
					    static_cast<Eigen::Index>(input1_shape[0]),
					    static_cast<Eigen::Index>(input1_shape[1])};
					Eigen::array<Eigen::Index, 1> out1_dims{
					    static_cast<Eigen::Index>(output1_shape[0])};
					Eigen::array<Eigen::Index, 2> out2_dims{
					    static_cast<Eigen::Index>(output2_shape[0]),
					    static_cast<Eigen::Index>(output2_shape[1])};

					Eigen::TensorMap<
					    Eigen::Tensor<ElementType, 1, Eigen::RowMajor>>
					out1(static_cast<ElementType*>(output1), out1_dims);
					Eigen::TensorMap<
					    Eigen::Tensor<ElementType, 2, Eigen::RowMajor>>
					out2(static_cast<ElementType*>(output2), out2_dims);
					Eigen::TensorMap<
					    Eigen::Tensor<ElementType, 2, Eigen::RowMajor>>
					in0(static_cast<ElementType*>(input0), in_dims);

					auto m =
					    MatrixType::Map(in0.data(), in_dims[0], in_dims[1]);

					Eigen::JacobiSVD<MatrixType> svd(
					    m, Eigen::ComputeThinU | Eigen::ComputeThinV);

					Eigen::TensorMap<
					    Eigen::Tensor<ElementType, 1, Eigen::RowMajor>>
					eval_tensor(
					    const_cast<ElementType*>(svd.singularValues().data()),
					    out1_dims);

					Eigen::TensorMap<
					    Eigen::Tensor<ElementType, 2, Eigen::RowMajor>>
					evec_tensor(
					    const_cast<ElementType*>(svd.matrixV().data()),
					    out2_dims);

					out1.device(ngraph::runtime::cpu::executor::GetCPUExecutor()
					                .get_device(arena)) = eval_tensor;

					out2.device(ngraph::runtime::cpu::executor::GetCPUExecutor()
					                .get_device(arena)) = evec_tensor;
				}
			} // namespace kernel
		}     // namespace cpu
	}         // namespace runtime
} // namespace ngraph

namespace ngraph
{
	namespace runtime
	{
		namespace reference
		{
			// template <typename T>
			// void jacobi_svd(const T* arg0, T* out0, T* out1, size_t count)
			// {
			// 	// ideally would just write the plain jacobi rotation algo here
			//  	ngraph::runtime::cpu::kernel::jacobi_svd<T>((void*)arg0,
			//  (void*)out0, (void*)out1, count, 0);
			// }
		}
	} // namespace runtime
} // namespace ngraph