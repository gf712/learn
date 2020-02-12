#include "mean.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/broadcast.hpp"

using namespace ngraph;

constexpr NodeTypeInfo op::MeanOp::type_info;

op::MeanOp::MeanOp(const Output<Node>& arg, const AxisSet& reduction_axes)
    : util::ArithmeticReduction(arg, reduction_axes)
{
	constructor_validate_and_infer_types();
}

op::MeanOp::MeanOp(const Output<Node>& arg, const Output<Node>& reduction_axes)
    : util::ArithmeticReduction(arg, reduction_axes)
{
	constructor_validate_and_infer_types();
}

std::shared_ptr<Node>
op::MeanOp::copy_with_new_args(const NodeVector& new_args) const
{
	check_new_args_count(this, new_args);
	return std::make_shared<op::MeanOp>(new_args.at(0), new_args.at(1));
}

void op::MeanOp::generate_adjoints(
    autodiff::Adjoints& adjoints, const OutputVector& deltas)
{
	auto delta = deltas.at(0);

	auto x = input_value(0);
	auto& x_shape = x.get_shape();

	adjoints.add_delta(
	    x,
	    std::make_shared<op::Broadcast>(delta, x_shape, get_reduction_axes()));
}

std::shared_ptr<Node> op::MeanOp::get_default_value() const
{
	return ngraph::make_constant_from_string(
	    "0", get_element_type(), get_shape());
}
