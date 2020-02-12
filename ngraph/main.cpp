#include "eigensolver.hpp"
#include "mean.hpp"

#include "ngraph/axis_set.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/op.hpp"

using namespace ngraph;
using namespace std;

namespace ngraph
{
	namespace runtime
	{
		namespace cpu
		{
			template <>
			void Builder::BUILDER_DECL(ngraph::op::MeanOp)
			{
				BUILD_REDUCTION_FUNCTOR(MeanOp, mean);
			}

			void register_builders_mean_cpp()
			{
				REGISTER_OP_BUILDER(MeanOp);
			}

			template <>
			void Builder::BUILDER_DECL(ngraph::op::JacobiSVD)
			{
				auto jacobi_svd =
				    static_cast<const ngraph::op::JacobiSVD*>(node);

				auto& functors = external_function->get_functors();

				std::function<void(
				    void*, void*, void*, const Shape&, const Shape&,
				    const Shape&, int)>
				    kernel;

				auto element_type = node->get_input_element_type(0);

				if (element_type == element::f32)
					kernel = kernel::jacobi_svd<float>;
				else
					throw std::runtime_error(
					    "JacobiSVD cannot handle provided type");

				auto arg0_shape = args[0].get_shape();
				auto result0_shape = out[0].get_shape();
				auto result1_shape = out[1].get_shape();

				auto arg0_buffer_index =
				    external_function->get_buffer_index(args[0].get_name());
				auto out0_buffer_index =
				    external_function->get_buffer_index(out[0].get_name());
				auto out1_buffer_index =
				    external_function->get_buffer_index(out[1].get_name());

				auto functor =
				    [&, kernel, arg0_shape, result0_shape, result1_shape,
				     arg0_buffer_index, out0_buffer_index, out1_buffer_index](
				        CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
					    kernel(
					        ctx->buffer_data[arg0_buffer_index],
					        ctx->buffer_data[out0_buffer_index],
					        ctx->buffer_data[out1_buffer_index], arg0_shape,
					        result0_shape, result1_shape, ectx->arena);
				    };
				functors.emplace_back(functor);
			}

			void register_builders_jacobi_svd_cpp()
			{
				REGISTER_OP_BUILDER(JacobiSVD);
			}
		} // namespace cpu
	}     // namespace runtime
} // namespace ngraph

std::shared_ptr<Node> mean(const std::shared_ptr<Node>& X)
{
	return std::make_shared<op::Sum>(X, AxisSet{1}) /
	       std::make_shared<op::Convert>(
	           std::make_shared<op::Broadcast>(
	               std::make_shared<op::Reshape>(
	                   std::make_shared<op::Slice>(
	                       std::make_shared<op::ShapeOf>(X), Coordinate{1},
	                       Coordinate{2}),
	                   AxisVector{0}, Shape{}),
	               Shape{2}, AxisSet{0}),
	           element::f32);
}

std::shared_ptr<Node> covariance(const std::shared_ptr<Node>& X)
{
	auto features_mean =
	    std::make_shared<op::Broadcast>(mean(X), X->get_shape(), AxisSet{1});

	auto centered_data = X - features_mean;

	return std::make_shared<op::MatMul>(
	    centered_data, centered_data, false, true);
}

int main()
{
	ngraph::runtime::cpu::register_builders_mean_cpp();
	ngraph::runtime::cpu::register_builders_jacobi_svd_cpp();

	// in column major
	Shape input_shape{2, 3};
	Shape output_shape1{2};
	Shape output_shape2{2, 2};

	// [
	//	[1, 2],
	//  [3, 4],
	//  [5, 6]
	// ]
	// in column major {{1,3,5},{2,4,6}}
	std::vector<float> X{1, 3, 5, 2, 4, 6};

	auto A = std::make_shared<op::Parameter>(element::f32, input_shape);

	// because this is column major and ngraph is row major
	// the reduction axis set is 0, instead of 1
	// auto column_mean = std::make_shared<op::MeanOp>(
	// 	std::make_shared<op::Transpose>(A, order), AxisSet{0});

	auto column_mean = mean(A);

	auto C = A - std::make_shared<op::Broadcast>(
	                 column_mean, A->get_shape(), AxisSet{1});

	auto V = covariance(C);

	auto jacobi_result = std::make_shared<op::JacobiSVD>(V);
	auto eigenvalues_op =
	    std::make_shared<op::GetOutputElement>(jacobi_result, 0);
	auto eigenvector_op =
	    std::make_shared<op::GetOutputElement>(jacobi_result, 1);

	auto f = std::make_shared<Function>(
	    OutputVector{eigenvalues_op, eigenvector_op}, ParameterVector{A});

	auto backend = runtime::Backend::create("CPU", true);
	auto a_tensor = backend->create_tensor(element::f32, input_shape);
	auto eigenvalue_tensor =
	    backend->create_tensor(element::f32, output_shape1);
	auto eigenvector_tensor =
	    backend->create_tensor(element::f32, output_shape2);

	a_tensor->write(X.data(), X.size() * sizeof(float));

	auto handle = backend->compile(f);
	handle->call_with_validate(
	    {eigenvalue_tensor, eigenvector_tensor}, {a_tensor});

	size_t eigenvalue_size = ngraph::shape_size(eigenvalue_tensor->get_shape());
	size_t eigenvector_size =
	    ngraph::shape_size(eigenvector_tensor->get_shape());

	std::vector<float> eigenvalues(eigenvalue_size);
	std::vector<float> eigenvectors(eigenvector_size);

	eigenvalue_tensor->read(
	    eigenvalues.data(), eigenvalue_size * sizeof(float));
	eigenvector_tensor->read(
	    eigenvectors.data(), eigenvector_size * sizeof(float));

	std::cout << "Eigenvalues: \n";
	for (const auto& el : eigenvalues)
		std::cout << el << ", ";
	std::cout << '\n';

	std::cout << "Eigenvectors: \n";
	for (const auto& el : eigenvectors)
		std::cout << el << ", ";
	std::cout << '\n';
}